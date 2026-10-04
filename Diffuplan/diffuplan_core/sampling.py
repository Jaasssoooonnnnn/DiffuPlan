"""Diffusion-style sampling helpers for LLaDA2 editable-plan inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .actions import TERMINAL_ACTION_CLOSE_TAGS, postprocess_to_first_action
from .plan_editor import (
    PlanEdit,
    apply_update_duplicate_replacements,
    char_spans_to_token_owners,
    update_spans_to_mask_slots,
)

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - lets parser-only smoke tests run without torch.
    torch = None
    F = None


@dataclass
class SamplingConfig:
    max_new_tokens: int = 256
    block_length: int = 32
    denoising_steps: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    confidence_threshold: float = 0.95
    mask_token_id: int = 156895
    eos_token_id: int = 156892
    expand_token_id: int = 156901
    expand_token_text: str = "<|reserved_token_6|>"
    expand_alias: str = "<expand>"
    max_expand_per_edit: int = 32
    edit_transfer_tokens: int = 1


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("torch is required for LLaDA2 sampling.")


def render_internal_tokens(text: str, cfg: SamplingConfig) -> str:
    return (text or "").replace(cfg.expand_token_text, cfg.expand_alias)


def _contains_mask_token(tokenizer: Any, text: str, cfg: SamplingConfig) -> bool:
    if not text:
        return False
    if "<|mask|>" in text or "<mask>" in text:
        return True
    return int(cfg.mask_token_id) in tokenizer.encode(text, add_special_tokens=False)


def _strip_forbidden_action_tokens(text: str, cfg: SamplingConfig) -> str:
    """Keep ordinary action text free of controller-only mask/expand artifacts."""

    cleaned = text or ""
    for item in (cfg.expand_token_text, cfg.expand_alias, "<|mask|>", "<mask>"):
        if item:
            cleaned = cleaned.replace(item, "")
    return cleaned


def _top_k_logits(logits: Any, k: int) -> Any:
    if not k or k <= 0:
        return logits
    values, _ = torch.topk(logits, min(k, logits.shape[-1]))
    return torch.where(logits < values[..., -1, None], torch.full_like(logits, float("-inf")), logits)


def _top_p_logits(logits: Any, p: float) -> Any:
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    sorted_mask = cumulative > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def _sample_token(
    logits: Any,
    cfg: SamplingConfig,
    *,
    allow_expand: bool,
    banned_token_ids: Optional[set[int]] = None,
) -> tuple[int, float]:
    _require_torch()
    work = logits.float().clone()
    banned = set(banned_token_ids or set())
    banned.add(int(cfg.mask_token_id))
    if not allow_expand:
        banned.add(int(cfg.expand_token_id))
    for token_id in banned:
        if 0 <= token_id < work.shape[-1]:
            work[token_id] = float("-inf")

    if cfg.temperature and cfg.temperature > 0:
        work = work / float(cfg.temperature)
        work = _top_k_logits(work, cfg.top_k)
        work = _top_p_logits(work, cfg.top_p)
        probs = F.softmax(work, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        prob = probs[token].item()
        return int(token.item()), float(prob)

    probs = F.softmax(work, dim=-1)
    token = torch.argmax(probs, dim=-1)
    return int(token.item()), float(probs[token].item())


def _render_token_id(tokenizer: Any, token_id: int, cfg: SamplingConfig) -> str:
    if int(token_id) == int(cfg.mask_token_id):
        return "<mask>"
    if int(token_id) == int(cfg.expand_token_id):
        return cfg.expand_alias
    try:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        text = f"<id:{int(token_id)}>"
    return render_internal_tokens(text, cfg)


def _get_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


def generate_action_text(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    cfg: SamplingConfig,
) -> tuple[str, int]:
    """Generate until a complete terminal action closing tag is seen."""

    _require_torch()
    device = _get_device(model)
    prefix_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    generated: list[int] = []
    total_budget = max(1, int(cfg.max_new_tokens))
    block_length = max(1, int(cfg.block_length))
    steps = max(1, int(cfg.denoising_steps))

    with torch.no_grad():
        while len(generated) < total_budget:
            current_block = min(block_length, total_budget - len(generated))
            block = [int(cfg.mask_token_id)] * current_block
            committed = [False] * current_block

            for _ in range(steps):
                if all(committed):
                    break
                input_ids = torch.tensor([prefix_ids + generated + block], dtype=torch.long, device=device)
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[0, -current_block:, :]

                active_positions = [idx for idx, done in enumerate(committed) if not done]
                if not active_positions:
                    break

                sampled: list[tuple[float, int, int]] = []
                for pos in active_positions:
                    token_id, prob = _sample_token(
                        logits[pos],
                        cfg,
                        allow_expand=False,
                        banned_token_ids={int(cfg.eos_token_id)},
                    )
                    sampled.append((prob, pos, token_id))

                threshold_hits = [item for item in sampled if item[0] >= cfg.confidence_threshold]
                if threshold_hits:
                    to_commit = threshold_hits
                else:
                    to_commit = [max(sampled, key=lambda item: item[0])]

                for _, pos, token_id in to_commit:
                    block[pos] = token_id
                    committed[pos] = True

                text = tokenizer.decode(generated + block, skip_special_tokens=False)
                if any(close_tag in text for close_tag in TERMINAL_ACTION_CLOSE_TAGS):
                    processed = postprocess_to_first_action(text)
                    if not _contains_mask_token(tokenizer, processed, cfg):
                        return processed, len(tokenizer.encode(processed, add_special_tokens=False))

            if not all(committed):
                input_ids = torch.tensor([prefix_ids + generated + block], dtype=torch.long, device=device)
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[0, -current_block:, :]
                for pos, done in enumerate(committed):
                    if done:
                        continue
                    token_id, _ = _sample_token(
                        logits[pos],
                        cfg,
                        allow_expand=False,
                        banned_token_ids={int(cfg.eos_token_id)},
                    )
                    block[pos] = token_id
                    committed[pos] = True

            generated.extend(block)
            text = tokenizer.decode(generated, skip_special_tokens=False)
            if any(close_tag in text for close_tag in TERMINAL_ACTION_CLOSE_TAGS):
                processed = postprocess_to_first_action(text)
                if not _contains_mask_token(tokenizer, processed, cfg):
                    return processed, len(tokenizer.encode(processed, add_special_tokens=False))

    text = tokenizer.decode(generated, skip_special_tokens=False)
    processed = postprocess_to_first_action(text)
    return processed, len(tokenizer.encode(processed, add_special_tokens=False))


def generate_action_text_dinfer(
    dinfer_engine: Any,
    tokenizer: Any,
    prompt_text: str,
    cfg: SamplingConfig,
    *,
    return_trace: bool = False,
) -> tuple[str, int, dict[str, Any]]:
    """Generate the next editable-plan action with dInfer append generation.

    Traces retain original token IDs, commit steps, and the full terminal block.
    Visible text may end inside that block; its remaining sampled tokens are
    conditioning only. The learner never re-encodes generated tokens.
    """

    raw_text, _, info = dinfer_engine.generate_text_once(
        prompt_text=prompt_text,
        cfg=cfg,
        stop_texts=TERMINAL_ACTION_CLOSE_TAGS,
        banned_token_ids={int(cfg.mask_token_id), int(cfg.expand_token_id)},
        return_step_map=bool(return_trace),
        return_token_ids=bool(return_trace),
    )
    return _finish_dinfer_action(raw_text, info, tokenizer, cfg, return_trace)


def generate_action_texts_dinfer(dinfer_engine, tokenizer, prompt_texts, cfg, *, return_trace=False):
    outputs = dinfer_engine.generate_text_batch_once(
        prompt_texts=prompt_texts, cfg=cfg, stop_texts=TERMINAL_ACTION_CLOSE_TAGS,
        banned_token_ids={int(cfg.mask_token_id), int(cfg.expand_token_id)},
        return_step_map=return_trace, return_token_ids=return_trace,
    )
    return [_finish_dinfer_action(text, info, tokenizer, cfg, return_trace)
            for text, _, info in outputs]


def _finish_dinfer_action(raw_text, info, tokenizer, cfg, return_trace):
    processed = postprocess_to_first_action(raw_text)
    processed = _strip_forbidden_action_tokens(processed, cfg)
    token_count = len(tokenizer.encode(processed, add_special_tokens=False)) if processed else 0
    info = dict(info or {})
    info.update(
        {
            "backend": "dinfer",
            "raw_generated_tokens": int(info.get("generated_tokens", 0) or 0),
            "generated_tokens": int(token_count),
        }
    )
    if return_trace:
        raw_step_map = list(info.get("step_map", []) or [])
        raw_token_ids = info.get("token_ids")
        if raw_token_ids is not None:
            # Use the engine's REAL generated token ids (parallel to step_map, EOS-trimmed;
            # mask/expand are banned at generation so never appear). postprocess_to_first_action
            # only trims a suffix at the first terminal close tag, so `processed` is a decoded
            # prefix of these tokens. Find the kept token count by aligning decoded length to
            # `processed` (binary search; decoded char-length is monotonic in token count). This
            # avoids re-encoding the text, which would re-tokenize differently (`>#` merge drift).
            raw_token_ids = [int(t) for t in raw_token_ids]
            target_len = len(processed or "")
            if target_len <= 0:
                kept = 0
            else:
                lo, hi = 0, len(raw_token_ids)
                while lo < hi:
                    mid = (lo + hi) // 2
                    decoded = tokenizer.decode(raw_token_ids[:mid], skip_special_tokens=False)
                    if len(decoded) >= target_len:
                        hi = mid
                    else:
                        lo = mid + 1
                kept = lo
            seg_ids = raw_token_ids[:kept]
            seg_step_map = list(raw_step_map[:kept])
        else:
            # transformers backend: no real token ids returned -> fall back to a length-only
            # re-encode (the drift-prone path, used only when token_ids is unavailable).
            seg_ids = tokenizer.encode(processed, add_special_tokens=False) if processed else []
            seg_step_map = list(raw_step_map[: len(seg_ids)])
        seg_len = len(seg_ids)
        if len(seg_step_map) != seg_len:
            raise ValueError("Action token IDs and diffusion steps must match; never invent missing steps")
        if info.get("trace_version") == 2:
            context_ids = info["conditioning_token_ids"]
            context_steps = info["conditioning_step_map"]
            if len(context_ids) != len(context_steps) or context_ids[:seg_len] != seg_ids:
                raise ValueError("Terminal-block replay is not aligned with the visible action")
            # Supervise visible sampled tokens and a sampled EOS that ended generation.
            # Other tokens in the terminal block are retained ONLY as conditioning.
            loss_end = seg_len
            if (seg_len == len(raw_token_ids) and seg_len < len(context_ids)
                    and context_ids[seg_len] == int(cfg.eos_token_id)
                    and context_steps[seg_len] >= 0):
                loss_end += 1
            info["action_loss_mask"] = [
                i < loss_end and step >= 0 and token not in {int(cfg.mask_token_id), int(cfg.expand_token_id)}
                for i, (token, step) in enumerate(zip(context_ids, context_steps))
            ]
        info["segment_step_map"] = seg_step_map
        info["segment_token_ids"] = [int(t) for t in seg_ids]
    return processed, token_count, info


def denoise_plan_edit(
    model: Any,
    tokenizer: Any,
    *,
    prefix_text: str,
    plan_inner: str,
    suffix_text: str,
    edit: PlanEdit,
    cfg: SamplingConfig,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Denoise editable plan masks by confidence, with expand-token handling."""

    _require_torch()
    if not edit.spans:
        return plan_inner, {}, {"masked_tokens": 0, "expanded": 0, "status": "no_spans"}

    device = _get_device(model)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    if edit.kind == "update":
        plan_ids, owners = update_spans_to_mask_slots(
            tokenizer,
            plan_inner,
            edit.spans,
            mask_token_id=int(cfg.mask_token_id),
        )
    else:
        plan_ids, owners = char_spans_to_token_owners(tokenizer, plan_inner, edit.spans)

    masked_count = sum(owner is not None for owner in owners)
    if masked_count == 0:
        return plan_inner, {}, {"masked_tokens": 0, "expanded": 0, "status": "no_token_overlap"}

    for idx, owner in enumerate(owners):
        if owner is not None:
            plan_ids[idx] = int(cfg.mask_token_id)

    expanded = 0
    steps = 0
    max_steps = max(1, len(plan_ids) + int(cfg.max_expand_per_edit) * 2 + 32)

    commit_order: list[int] = []
    edit_events: list[dict[str, Any]] = []

    with torch.no_grad():
        while any(owner is not None and token_id == cfg.mask_token_id for token_id, owner in zip(plan_ids, owners)):
            if steps >= max_steps:
                break
            steps += 1
            input_ids = torch.tensor([prefix_ids + plan_ids + suffix_ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, len(prefix_ids) : len(prefix_ids) + len(plan_ids), :]
            active_positions = [
                idx
                for idx, (token_id, owner) in enumerate(zip(plan_ids, owners))
                if owner is not None and token_id == cfg.mask_token_id
            ]
            if not active_positions:
                break

            sampled: list[tuple[float, int, int]] = []
            for local_pos in active_positions:
                banned = {int(cfg.eos_token_id)}
                if expanded >= int(cfg.max_expand_per_edit):
                    banned.add(int(cfg.expand_token_id))
                token_id, prob = _sample_token(
                    logits[local_pos],
                    cfg,
                    allow_expand=True,
                    banned_token_ids=banned,
                )
                sampled.append((prob, local_pos, token_id))

            transfer_count = max(1, int(cfg.edit_transfer_tokens))
            to_commit = sorted(sampled, key=lambda item: item[0], reverse=True)[:transfer_count]

            # Mutate from right to left so earlier positions are not shifted.
            for prob, local_pos, token_id in sorted(to_commit, key=lambda item: item[1], reverse=True):
                if owners[local_pos] is None or plan_ids[local_pos] != int(cfg.mask_token_id):
                    continue
                owner = owners[local_pos]
                plan_len_before = len(plan_ids)
                masks_before = sum(
                    1
                    for current_id, current_owner in zip(plan_ids, owners)
                    if current_owner is not None and int(current_id) == int(cfg.mask_token_id)
                )
                commit_order.append(int(local_pos))
                if token_id == int(cfg.expand_token_id) and expanded < int(cfg.max_expand_per_edit):
                    plan_ids[local_pos : local_pos + 1] = [int(cfg.mask_token_id), int(cfg.mask_token_id)]
                    owners[local_pos : local_pos + 1] = [owner, owner]
                    expanded += 1
                    edit_events.append(
                        {
                            "step": int(steps),
                            "action": "expand",
                            "local_pos": int(local_pos),
                            "owner": owner,
                            "token_id": int(token_id),
                            "token_text": _render_token_id(tokenizer, int(token_id), cfg),
                            "prob": float(prob),
                            "plan_len_before": int(plan_len_before),
                            "plan_len_after": int(len(plan_ids)),
                            "masks_before": int(masks_before),
                            "masks_after": int(
                                sum(
                                    1
                                    for current_id, current_owner in zip(plan_ids, owners)
                                    if current_owner is not None and int(current_id) == int(cfg.mask_token_id)
                                )
                            ),
                        }
                    )
                    continue

                plan_ids[local_pos] = int(token_id)
                edit_events.append(
                    {
                        "step": int(steps),
                        "action": "fill",
                        "local_pos": int(local_pos),
                        "owner": owner,
                        "token_id": int(token_id),
                        "token_text": _render_token_id(tokenizer, int(token_id), cfg),
                        "prob": float(prob),
                        "plan_len_before": int(plan_len_before),
                        "plan_len_after": int(len(plan_ids)),
                        "masks_before": int(masks_before),
                        "masks_after": int(
                            sum(
                                1
                                for current_id, current_owner in zip(plan_ids, owners)
                                if current_owner is not None and int(current_id) == int(cfg.mask_token_id)
                            )
                        ),
                    }
                )

    new_plan_inner = tokenizer.decode(plan_ids, skip_special_tokens=False)
    replacements: dict[str, str] = {}
    for owner in sorted({owner for owner in owners if owner}):
        if not owner.startswith("#A_"):
            continue
        owner_ids = [token_id for token_id, token_owner in zip(plan_ids, owners) if token_owner == owner]
        replacement = tokenizer.decode(owner_ids, skip_special_tokens=False)
        replacement = render_internal_tokens(replacement, cfg).strip()
        replacements[owner] = replacement

    if edit.kind == "update" and replacements:
        new_plan_inner = apply_update_duplicate_replacements(new_plan_inner, replacements)

    info = {
        "masked_tokens": int(masked_count),
        "expanded": int(expanded),
        "steps": int(steps),
        "edit_transfer_tokens": int(cfg.edit_transfer_tokens),
        "commit_order": commit_order,
        "edit_events": edit_events,
        "status": "complete" if steps < max_steps else "max_steps",
    }
    return new_plan_inner, replacements, info


def _sample_tokens_device(logits, cfg, *, banned_token_ids):
    """Retain each edit request's original sampling order and tensor shapes."""
    work = logits.float().clone()
    banned = sorted(set(banned_token_ids) | {int(cfg.mask_token_id)})
    banned = [value for value in banned if 0 <= value < work.shape[-1]]
    work[:, banned] = float("-inf")
    if cfg.temperature and cfg.temperature > 0:
        work = _top_p_logits(_top_k_logits(work / float(cfg.temperature), cfg.top_k), cfg.top_p)
        probs = F.softmax(work, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1)
    else:
        probs = F.softmax(work, dim=-1)
        tokens = probs.argmax(dim=-1, keepdim=True)
    confidence = probs.gather(-1, tokens)
    return tokens, confidence


def _sample_tokens(logits, cfg, *, banned_token_ids):
    tokens, confidence = _sample_tokens_device(logits, cfg, banned_token_ids=banned_token_ids)
    return list(zip(tokens.flatten().tolist(), confidence.flatten().tolist()))


@torch.no_grad()
def sample_edit_choices_batch(logits, requests):
    """Queue unchanged per-request sampling, then transfer all choices together."""
    payloads = []
    lengths = []
    for values, request in zip(logits, requests):
        tokens, confidence = _sample_tokens_device(
            values, request["cfg"], banned_token_ids=request["banned_token_ids"]
        )
        # Vocabulary IDs are exactly representable in float32 for this model.
        if values.shape[-1] > 2**24:
            raise ValueError("Combined edit payload requires vocabulary <= 2**24")
        payloads.append(torch.cat((tokens.float(), confidence), dim=-1))
        lengths.append(len(values))
    host = torch.cat(payloads).cpu().tolist()
    result = []
    start = 0
    for length in lengths:
        result.append([(int(token), float(prob)) for token, prob in host[start:start + length]])
        start += length
    return result


def execute_edit_logits_request(request):
    engine = request["engine"]
    if getattr(engine, "batch_edit_sampling", False) or getattr(engine, "optimize_decode", False):
        return engine.forward_edit_logits_batch([request])[0]
    with torch.no_grad():
        inputs = torch.tensor([request["canvas_ids"]], dtype=torch.long, device=engine.device)
        output = engine.model_runner.forward(input_ids=inputs)
        logits = getattr(output, "logits", output)
        return logits[0, request["positions"], :]


def denoise_plan_edit_dinfer(*args, **kwargs):
    steps = denoise_plan_edit_steps(*args, **kwargs)
    response = None
    while True:
        try:
            kind, request = steps.send(response)
        except StopIteration as done:
            return done.value
        response = execute_edit_logits_request(request)


def denoise_plan_edit_steps(
    dinfer_engine: Any,
    tokenizer: Any,
    *,
    prefix_text: str,
    plan_inner: str,
    suffix_text: str,
    edit: PlanEdit,
    cfg: SamplingConfig,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Denoise editable plan masks with the dInfer diffusion runner.

    This mirrors ``denoise_plan_edit``'s span/expand semantics, but obtains each
    masked-canvas fill pass from dInfer's SGLang ``model_runner.forward``. The
    append-only ``BlockDiffusionLLM.generate`` path does not fill masks already
    present inside the prompt, so plan edits need this direct canvas forward.
    """

    _require_torch()
    if not edit.spans:
        return plan_inner, {}, {"masked_tokens": 0, "expanded": 0, "status": "no_spans", "backend": "dinfer"}
    # Post-sync storage-rebind tears down runtime modules (model_runner=None). The action
    # path lazily rebuilds via _ensure_ready_for_generate; the edit path has no such hook,
    # so rebuild here before requiring model_runner (otherwise the first post-sync edit raises).
    if getattr(dinfer_engine, "model_runner", None) is None:
        ensure_ready = getattr(dinfer_engine, "_ensure_ready_for_generate", None)
        if callable(ensure_ready):
            ensure_ready()
    if getattr(dinfer_engine, "model_runner", None) is None or getattr(dinfer_engine, "device", None) is None:
        raise RuntimeError("Initialized dInfer engine with model_runner/device is required for plan edit denoising.")

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    if edit.kind == "update":
        plan_ids, owners = update_spans_to_mask_slots(
            tokenizer,
            plan_inner,
            edit.spans,
            mask_token_id=int(cfg.mask_token_id),
        )
    else:
        plan_ids, owners = char_spans_to_token_owners(tokenizer, plan_inner, edit.spans)

    masked_count = sum(owner is not None for owner in owners)
    if masked_count == 0:
        return plan_inner, {}, {"masked_tokens": 0, "expanded": 0, "status": "no_token_overlap", "backend": "dinfer"}

    for idx, owner in enumerate(owners):
        if owner is not None:
            plan_ids[idx] = int(cfg.mask_token_id)

    expanded = 0
    steps = 0
    commit_order: list[int] = []
    edit_events: list[dict[str, Any]] = []
    max_steps = max(1, len(plan_ids) + int(cfg.max_expand_per_edit) * 2 + 32)

    device = dinfer_engine.device
    prefix_len = len(prefix_ids)

    while any(owner is not None and token_id == int(cfg.mask_token_id) for token_id, owner in zip(plan_ids, owners)):
        if steps >= max_steps:
            break
        steps += 1
        canvas_ids = prefix_ids + plan_ids + suffix_ids
        active_positions = [
            idx
            for idx, (token_id, owner) in enumerate(zip(plan_ids, owners))
            if owner is not None and token_id == int(cfg.mask_token_id)
        ]
        if not active_positions:
            break

        banned = {int(cfg.eos_token_id)}
        if expanded >= int(cfg.max_expand_per_edit):
            banned.add(int(cfg.expand_token_id))
        combined_host_transfer = (
            getattr(dinfer_engine, "optimize_decode", False)
            and getattr(dinfer_engine, "batch_edit_sampling", False)
        )
        response = yield ("edit_logits", {
            "engine": dinfer_engine, "canvas_ids": canvas_ids,
            "positions": [prefix_len + pos for pos in active_positions],
            "cfg": cfg, "banned_token_ids": sorted(banned),
            "return_choices": combined_host_transfer,
        })
        if combined_host_transfer:
            choices = response
        elif getattr(dinfer_engine, "batch_edit_sampling", False):
            choices = _sample_tokens(response, cfg, banned_token_ids=banned)
        else:
            choices = [_sample_token(row, cfg, allow_expand=True, banned_token_ids=banned)
                       for row in response]
        sampled = [(prob, pos, token) for pos, (token, prob) in zip(active_positions, choices)]

        transfer_count = max(1, int(cfg.edit_transfer_tokens))
        to_commit = sorted(sampled, key=lambda item: item[0], reverse=True)[:transfer_count]

        committed_this_round = 0
        for prob, local_pos, token_id in sorted(to_commit, key=lambda item: item[1], reverse=True):
            if owners[local_pos] is None or plan_ids[local_pos] != int(cfg.mask_token_id):
                continue
            owner = owners[local_pos]
            plan_len_before = len(plan_ids)
            masks_before = sum(
                1
                for current_id, current_owner in zip(plan_ids, owners)
                if current_owner is not None and int(current_id) == int(cfg.mask_token_id)
            )
            commit_order.append(int(local_pos))
            if token_id == int(cfg.expand_token_id) and expanded < int(cfg.max_expand_per_edit):
                plan_ids[local_pos : local_pos + 1] = [int(cfg.mask_token_id), int(cfg.mask_token_id)]
                owners[local_pos : local_pos + 1] = [owner, owner]
                expanded += 1
                committed_this_round += 1
                edit_events.append(
                    {
                        "step": int(steps),
                        "action": "expand",
                        "local_pos": int(local_pos),
                        "owner": owner,
                        "token_id": int(token_id),
                        "token_text": _render_token_id(tokenizer, int(token_id), cfg),
                        "prob": float(prob),
                        "plan_len_before": int(plan_len_before),
                        "plan_len_after": int(len(plan_ids)),
                        "masks_before": int(masks_before),
                        "masks_after": int(
                            sum(
                                1
                                for current_id, current_owner in zip(plan_ids, owners)
                                if current_owner is not None and int(current_id) == int(cfg.mask_token_id)
                            )
                        ),
                    }
                )
                continue
            if token_id == int(cfg.expand_token_id):
                edit_events.append(
                    {
                        "step": int(steps),
                        "action": "skip_expand",
                        "local_pos": int(local_pos),
                        "owner": owner,
                        "token_id": int(token_id),
                        "token_text": _render_token_id(tokenizer, int(token_id), cfg),
                        "prob": float(prob),
                        "plan_len_before": int(plan_len_before),
                        "plan_len_after": int(len(plan_ids)),
                        "masks_before": int(masks_before),
                        "masks_after": int(masks_before),
                    }
                )
                continue
            plan_ids[local_pos] = token_id
            committed_this_round += 1
            edit_events.append(
                {
                    "step": int(steps),
                    "action": "fill",
                    "local_pos": int(local_pos),
                    "owner": owner,
                    "token_id": int(token_id),
                    "token_text": _render_token_id(tokenizer, int(token_id), cfg),
                    "prob": float(prob),
                    "plan_len_before": int(plan_len_before),
                    "plan_len_after": int(len(plan_ids)),
                    "masks_before": int(masks_before),
                    "masks_after": int(
                        sum(
                            1
                            for current_id, current_owner in zip(plan_ids, owners)
                            if current_owner is not None and int(current_id) == int(cfg.mask_token_id)
                        )
                    ),
                }
            )

        if committed_this_round == 0:
            break


    new_plan_inner = tokenizer.decode(plan_ids, skip_special_tokens=False)
    replacements: dict[str, str] = {}
    for owner in sorted({owner for owner in owners if owner}):
        if not owner.startswith("#A_"):
            continue
        owner_ids = [token_id for token_id, token_owner in zip(plan_ids, owners) if token_owner == owner]
        replacement = tokenizer.decode(owner_ids, skip_special_tokens=False)
        replacement = render_internal_tokens(replacement, cfg).strip()
        replacements[owner] = replacement

    if edit.kind == "update" and replacements:
        new_plan_inner = apply_update_duplicate_replacements(new_plan_inner, replacements)

    remaining_masks = sum(
        1
        for token_id, owner in zip(plan_ids, owners)
        if owner is not None and int(token_id) == int(cfg.mask_token_id)
    )
    if remaining_masks:
        status = "max_steps" if steps >= max_steps else "stalled"
    else:
        status = "complete"

    info = {
        "backend": "dinfer",
        "masked_tokens": int(masked_count),
        "remaining_masks": int(remaining_masks),
        "expanded": int(expanded),
        "steps": int(steps),
        "edit_transfer_tokens": int(cfg.edit_transfer_tokens),
        "commit_order": commit_order,
        "edit_events": edit_events,
        "status": status,
        # TraceRL edit trace (TOKEN space, no char re-tokenization). The edit-RL loss
        # rebuilds the fully-masked canvas from these: prefix_ids + [final plan_ids, with
        # each owned (masked-then-denoised) run re-masked one-mask-per-token] + suffix_ids,
        # with the denoised owned tokens as the target R. owner_per_pos marks which final
        # plan positions were produced by the edit (the supervised region).
        "edit_trace": {
            "prefix_ids": [int(t) for t in prefix_ids],
            "suffix_ids": [int(t) for t in suffix_ids],
            "final_plan_ids": [int(t) for t in plan_ids],
            "owner_per_pos": [owner for owner in owners],
            "mask_token_id": int(cfg.mask_token_id),
            "expand_token_id": int(cfg.expand_token_id),
            "remaining_masks": int(remaining_masks),
        },
    }
    return new_plan_inner, replacements, info
