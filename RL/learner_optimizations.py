"""Sparse action log-probability heads and grouped reference MoE, installed before FSDP."""
from types import MethodType

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def selected_linear_logp(hidden, weight, labels, chunk_size=128, *,
                         temperature=1.0, banned_token_ids=()):
    """Project only supervised positions; bound vocabulary intermediates.

    The FP32 log-softmax matches the original model's logits.float(). Non-reentrant
    checkpointing recomputes the small head chunks in backward rather than retaining
    a full vocabulary distribution for every selected token.
    """
    def project(h, w, y):
        from action_policy import token_logp
        logits = F.linear(h, w).float()
        return token_logp(logits, y, temperature=temperature, banned_token_ids=banned_token_ids)

    values = []
    for start in range(0, hidden.shape[0], max(1, int(chunk_size))):
        h = hidden[start:start + chunk_size]
        y = labels[start:start + chunk_size]
        if torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad):
            values.append(checkpoint(project, h, weight, y, use_reentrant=False))
        else:
            values.append(project(h, weight, y))
    if not values:
        # Keep the same FSDP forward/backward participation on padding ranks.
        return (hidden.sum() + weight.reshape(-1)[0]).reshape(1)[:0].float()
    return torch.cat(values)


def install_learner_optimizations(model, *, sparse_action_logits=True,
                                  grouped_reference_moe=True, logp_chunk_size=128):
    if sparse_action_logits:
        original_forward = model.forward

        def forward(self, *args, rl_token_indices=None, rl_token_labels=None,
                    rl_sampling_temperature=1.0, rl_banned_token_ids=(), **kwargs):
            if rl_token_indices is None:
                return original_forward(*args, **kwargs)
            if args or kwargs.get("labels") is not None:
                raise ValueError("Sparse RL forward requires keyword inputs without LM labels")
            decoder_kwargs = dict(kwargs)
            decoder_kwargs.update(return_dict=True, use_cache=False)
            outputs = self.model(**decoder_kwargs)
            hidden = outputs[0].reshape(-1, outputs[0].shape[-1])
            selected = hidden.index_select(0, rl_token_indices)
            logp = selected_linear_logp(selected, self.lm_head.weight,
                                       rl_token_labels, logp_chunk_size,
                                       temperature=rl_sampling_temperature,
                                       banned_token_ids=rl_banned_token_ids)
            return {"selected_logp": logp}

        model.forward = MethodType(forward, model)
        model._rl_sparse_logp = True

    if __import__("os").environ.get("RL_REPORT_KERNEL_DTYPES") == "1":
        for module in model.modules():
            if module.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
                original_triton = module._forward_training_triton
                seen = set()
                def trace_triton(self, hidden, *args, _original=original_triton, **kwargs):
                    mode = "train" if self.training else "reference"
                    if mode not in seen:
                        weight = self.experts[0].gate_proj.weight
                        print(f"[Kernel dtype] {mode} hidden={hidden.dtype} expert_weight={weight.dtype} "
                              f"shape={tuple(hidden.shape)}", flush=True)
                        seen.add(mode)
                    return _original(hidden, *args, **kwargs)
                module._forward_training_triton = MethodType(trace_triton, module)
                break

    patched = 0
    if grouped_reference_moe:
        for module in model.modules():
            if module.__class__.__name__ != "LLaDA2MoeSparseMoeBlock":
                continue
            original = module.forward

            def forward(self, hidden_states, _original=original):
                if self.training or hidden_states.device.type != "cuda":
                    return _original(hidden_states)
                # Run grouped MoE with the reference model in evaluation mode.
                indices, weights, router = self.gate(hidden_states)
                result = self._forward_training_triton(hidden_states, indices, weights)
                if self.config.num_shared_experts is not None:
                    result = result + self.shared_experts(hidden_states)
                b, n, _ = hidden_states.shape
                return result, (router.view(b, n, -1), indices.view(b, n, -1))

            module.forward = MethodType(forward, module)
            patched += 1
    return {"sparse_action_logits": bool(sparse_action_logits),
            "grouped_reference_layers": patched, "logp_chunk_size": int(logp_chunk_size)}
