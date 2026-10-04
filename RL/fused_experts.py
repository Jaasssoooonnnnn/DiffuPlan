"""Pack expert parameters while retaining the original HF checkpoint schema."""
import sys
from types import MethodType
import torch


class _RouteGather(torch.autograd.Function):
    """Avoid materializing the repeated token matrix before expert sorting."""
    @staticmethod
    def forward(ctx, hidden, order, topk):
        ctx.save_for_backward(order)
        ctx.topk = topk
        ctx.shape = hidden.shape
        return hidden.index_select(0, order // topk)

    @staticmethod
    def backward(ctx, grad_sorted):
        (order,) = ctx.saved_tensors
        # Match the original permutation + repeat_interleave backward, including
        # its reduction order, instead of BF16 atomic adds into duplicate tokens.
        restored = torch.empty_like(grad_sorted)
        restored[order] = grad_sorted
        return restored.view(ctx.shape[0], ctx.topk, ctx.shape[1]).sum(1), None, None


def _forward_fused(self, hidden_states, topk_idx, topk_weight):
    b, n, h = hidden_states.shape
    topk = self.num_experts_per_tok
    flat = hidden_states.reshape(-1, h)
    indices = topk_idx.reshape(-1)
    order = indices.argsort()
    sorted_hidden = _RouteGather.apply(flat, order, topk)
    # IDs are bounded by the router. Avoid bincount's GPU maximum -> host
    # shape read even though the number of experts is already known.
    counts = indices.new_zeros(self.config.num_experts)
    counts.scatter_add_(0, indices, torch.ones_like(indices))
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).contiguous()
    linear = sys.modules[self.__class__.__module__].triton_grouped_linear
    gate_up = linear(sorted_hidden, self.w1, offsets)
    if getattr(self.config, 'hidden_act', '') == 'silu':
        from moe_tensor_ops import swiglu
        intermediate = swiglu(gate_up)
    else:
        intermediate = self._rl_expert_activation(gate_up[:, :self._rl_intermediate]) * gate_up[:, self._rl_intermediate:]
    sorted_output = linear(intermediate, self.w2, offsets)
    from routed_combine import combine
    output = combine(sorted_output, order, topk_weight)
    return output.to(hidden_states.dtype).view(b, n, h)


def _forward(self, hidden_states):
    ids, weights, router = self.gate(hidden_states)
    out = self._forward_training_triton(hidden_states, ids, weights)
    if self.config.num_shared_experts is not None:
        out = out + self.shared_experts(hidden_states)
    b, n, _ = hidden_states.shape
    return out, (router.view(b, n, -1), ids.view(b, n, -1))


@torch.no_grad()
def fuse_expert_parameters(model):
    before = sum(p.numel() for p in model.parameters())
    count_before = len(list(model.parameters()))
    layers = 0
    for module in list(model.modules()):
        if module.__class__.__name__ != 'LLaDA2MoeSparseMoeBlock' or hasattr(module, 'w1'):
            continue
        experts = module.experts
        w1 = torch.stack([torch.cat((expert.gate_proj.weight, expert.up_proj.weight), 0) for expert in experts])
        w2 = torch.stack([expert.down_proj.weight for expert in experts])
        module._rl_intermediate = experts[0].gate_proj.weight.shape[0]
        module._rl_expert_activation = experts[0].act_fn
        del module.experts
        module.w1 = torch.nn.Parameter(w1)
        module.w2 = torch.nn.Parameter(w2)
        module._forward_training_triton = MethodType(_forward_fused, module)
        module.forward = MethodType(_forward, module)
        layers += 1
    if sum(p.numel() for p in model.parameters()) != before:
        raise RuntimeError('Fused expert conversion changed the parameter count')
    return dict(layers=layers, parameter_tensors_before=count_before,
                parameter_tensors_after=len(list(model.parameters())), total_parameters=before)


def unfuse_checkpoint_state(state):
    """Return independent CPU tensors in the original portable, unfused schema."""
    result = {}
    for name, value in state.items():
        if name.endswith('mlp.w1'):
            prefix = name[:-len('w1')] + 'experts.'
            intermediate = value.shape[1] // 2
            for expert in range(value.shape[0]):
                result[f'{prefix}{expert}.gate_proj.weight'] = value[expert, :intermediate].clone()
                result[f'{prefix}{expert}.up_proj.weight'] = value[expert, intermediate:].clone()
        elif name.endswith('mlp.w2'):
            prefix = name[:-len('w2')] + 'experts.'
            for expert in range(value.shape[0]):
                result[f'{prefix}{expert}.down_proj.weight'] = value[expert].clone()
        else:
            result[name] = value
    return result
