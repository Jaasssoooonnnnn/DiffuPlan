"""Editable-plan think/plan/search/update/replan/answer rollout loop."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .actions import Action, extract_latest_plan_inner, parse_action
from .plan_editor import build_plan_edit, find_latest_plan_span, replace_latest_plan_inner
from .prompting import (
    build_answer_direct_observation,
    build_information_observation,
    build_invalid_action_observation,
    build_prompt,
)
from .sampling import (
    SamplingConfig,
    denoise_plan_edit,
    denoise_plan_edit_dinfer,
    denoise_plan_edit_steps,
    execute_edit_logits_request,
    generate_action_text,
    generate_action_text_dinfer,
    generate_action_texts_dinfer,
    render_internal_tokens,
)
from .search_client import SearchClient


@dataclass
class RolloutConfig:
    max_turns: int = 12
    max_search_turns: int = 5
    max_obs_tokens: int = 500
    recovery_enabled: bool = True
    invalid_action_retries: int = 1
    answer_direct_retries: int = 1
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


@dataclass
class RolloutResult:
    question: str
    prompt: str
    full_output: str
    final_answer: str
    latest_plan: str
    actions: list[dict[str, Any]]
    retrieval_used: int
    generated_tokens_model_only: int
    stop_reason: str


def _execute_request(request):
    kind, kwargs = request
    if kind == "action":
        return generate_action_text_dinfer(**kwargs)
    if kind == "search":
        return kwargs["client"].search(kwargs["query"])
    if kind == "edit":
        return _apply_plan_edit(**kwargs)
    if kind == "edit_logits":
        return execute_edit_logits_request(kwargs)
    raise ValueError(kind)


def run_editable_plan_rollout(**kwargs):
    """Drive the shared controller sequentially (backwards-compatible API)."""
    steps = _editable_plan_steps(**kwargs)
    response = None
    while True:
        try:
            request = steps.send(response)
        except StopIteration as done:
            return done.value
        response = _execute_request(request)


def run_editable_plan_rollouts(*, episode_kwargs, batch_size, on_done=None):
    """Batch ready action requests from independent editable-plan controllers.

    Each controller keeps the identical full history, recovery, search, edit and
    stopping logic used by the sequential API. Results retain input order.
    Random samples can differ when the GPU batch/scheduling changes.
    """
    streams = [_editable_plan_steps(**kw) for kw in episode_kwargs]
    results = [None] * len(streams)
    pending = {}

    def advance(i, value=None):
        try:
            pending[i] = streams[i].send(value)
        except StopIteration as done:
            results[i] = done.value
            pending.pop(i, None)
            if on_done:
                on_done(i, done.value)
        except Exception as exc:
            # Preserve per-episode failure accounting, including failed edits.
            results[i] = exc
            pending.pop(i, None)
            if on_done:
                on_done(i, exc)

    for i in range(len(streams)):
        advance(i)
    while pending:
        # Complete ready environment/controller actions before choosing the next
        # GPU batch; grouping searches uses the real batched FAISS service.
        searches = [i for i, (kind, _) in pending.items() if kind == "search"]
        if searches:
            client = pending[searches[0]][1]["client"]
            values = client.batch_search([pending[i][1]["query"] for i in searches])
            if len(values) != len(searches):
                raise RuntimeError("Retriever returned the wrong batch size")
            for i, value in zip(searches, values):
                advance(i, value)
        edits = [i for i, (kind, _) in pending.items() if kind == "edit_logits"]
        edits.sort(key=lambda i: len(pending[i][1]["canvas_ids"]))
        for start in range(0, len(edits), max(1, int(batch_size))):
            ids = edits[start:start + max(1, int(batch_size))]
            requests = [pending[i][1] for i in ids]
            engine = requests[0]["engine"]
            if getattr(engine, "batch_edit_sampling", False):
                values = engine.forward_edit_logits_batch(requests)
            else:
                values = [execute_edit_logits_request(request) for request in requests]
            for i, value in zip(ids, values):
                advance(i, value)
        actions = [i for i, (kind, _) in pending.items() if kind == "action"]
        # Nearby history lengths reduce masked padding/prefill waste. All ready
        # episodes are eventually advanced; no rollout is dropped for speed.
        actions.sort(key=lambda i: len(pending[i][1]["prompt_text"]))
        start = 0
        while start < len(actions):
            remaining = min(max(1, int(batch_size)), len(actions) - start)
            size = remaining if remaining == batch_size else 2 ** (remaining.bit_length() - 1)
            ids = actions[start:start + size]
            start += size
            first = pending[ids[0]][1]
            outputs = generate_action_texts_dinfer(
                first["dinfer_engine"], first["tokenizer"],
                [pending[i][1]["prompt_text"] for i in ids], first["cfg"],
                return_trace=first["return_trace"],
            )
            for i, value in zip(ids, outputs):
                advance(i, value)
    return results


def _editable_plan_steps(
    *,
    model: Any,
    tokenizer: Any,
    question: str,
    search_client: Optional[SearchClient],
    config: RolloutConfig,
    action_dinfer_engine: Any = None,
    edit_dinfer_engine: Any = None,
    collect_trace: bool = False,
) -> RolloutResult:
    prompt = build_prompt(question)
    current_text = prompt
    actions: list[dict[str, Any]] = []
    retrieval_used = 0
    generated_tokens = 0
    stop_reason = "max_turns"
    final_answer_text = ""
    latest_plan_text = ""
    normal_turns_used = 0
    invalid_retries_used = 0
    answer_direct_retries_used = 0
    pending_invalid_recovery = False
    pending_answer_direct_reason: str | None = None
    max_turns = max(1, int(config.max_turns))
    recovery_enabled = bool(config.recovery_enabled)
    invalid_action_retries = max(0, int(config.invalid_action_retries))
    answer_direct_retries = max(0, int(config.answer_direct_retries))

    while True:
        if pending_answer_direct_reason is None and not pending_invalid_recovery and normal_turns_used >= max_turns:
            if recovery_enabled and answer_direct_retries_used < answer_direct_retries:
                current_text += build_answer_direct_observation()
                answer_direct_retries_used += 1
                pending_answer_direct_reason = "max_turns"
            else:
                stop_reason = "max_turns"
                break

        # The current history text is exactly this segment's frozen prefix (L0) for
        # per-action-segment RL training rows. Capture it before appending the segment.
        segment_prefix_text = current_text
        if action_dinfer_engine is not None:
            generated_text, token_count, generation_info = yield ("action", {
                "dinfer_engine": action_dinfer_engine, "tokenizer": tokenizer,
                "prompt_text": current_text, "cfg": config.sampling,
                "return_trace": bool(collect_trace),
            })
        else:
            generated_text, token_count = generate_action_text(
                model=model,
                tokenizer=tokenizer,
                prompt_text=current_text,
                cfg=config.sampling,
            )
            generation_info = {"backend": "transformers", "generated_tokens": int(token_count)}
        generated_tokens += int(token_count)
        current_text += generated_text
        rendered_generated = render_internal_tokens(generated_text, config.sampling)
        action = parse_action(generated_text)
        generated_plan = extract_latest_plan_inner(generated_text)
        if generated_plan.strip():
            latest_plan_text = generated_plan

        action_record: dict[str, Any] = {
            "turn": len(actions),
            "raw": rendered_generated,
            "kind": action.kind if action else None,
            "content": action.content if action else "",
            "generation": generation_info,
        }
        if collect_trace:
            # Per-segment action trace for TraceRL rows: frozen prefix + segment tokens + step_map.
            action_record["trace"] = {
                "segment_text": generated_text,
                "prefix_text": segment_prefix_text,
                **{key: generation_info[key] for key in (
                    "trace_version", "prompt_token_ids", "conditioning_token_ids",
                    "conditioning_step_map", "action_loss_mask", "sampling_temperature",
                    "banned_token_ids") if key in generation_info},
                "segment_token_ids": list(generation_info.get("segment_token_ids", []) or []),
                "segment_step_map": list(generation_info.get("segment_step_map", []) or []),
            }
        is_answer_direct_attempt = pending_answer_direct_reason is not None
        if not is_answer_direct_attempt and not pending_invalid_recovery:
            normal_turns_used += 1
        pending_invalid_recovery = False

        if action is None:
            action_record["error"] = "no_complete_action"
            actions.append(action_record)
            if is_answer_direct_attempt:
                stop_reason = pending_answer_direct_reason or "max_turns"
                break
            if recovery_enabled and invalid_retries_used < invalid_action_retries:
                current_text += build_invalid_action_observation()
                invalid_retries_used += 1
                pending_invalid_recovery = True
                continue
            stop_reason = "no_complete_action"
            break

        if is_answer_direct_attempt:
            actions.append(action_record)
            if action.kind == "answer":
                final_answer_text = action.content
                stop_reason = "answer"
            else:
                action_record["error"] = "answer_direct_expected"
                stop_reason = pending_answer_direct_reason or "max_turns"
            break

        if action.kind in {"think", "plan"}:
            actions.append(action_record)
            continue

        if action.kind == "answer":
            actions.append(action_record)
            final_answer_text = action.content
            stop_reason = "answer"
            break

        if action.kind == "search":
            if retrieval_used >= int(config.max_search_turns):
                action_record["error"] = "search_limit"
                actions.append(action_record)
                if recovery_enabled and answer_direct_retries_used < answer_direct_retries:
                    current_text += build_answer_direct_observation()
                    answer_direct_retries_used += 1
                    pending_answer_direct_reason = "search_limit"
                    continue
                stop_reason = "search_limit"
                break
            query = normalize_search_query(action.content)
            info_text = ""
            if search_client is not None:
                info_text = yield ("search", {"client": search_client, "query": query})
            info_text = _truncate_by_tokens(tokenizer, info_text, int(config.max_obs_tokens))
            current_text += build_information_observation(info_text)
            retrieval_used += 1
            action_record["query"] = query
            action_record["observation"] = info_text
            actions.append(action_record)
            continue

        if action.kind in {"update", "replan"}:
            edit_record = yield from _apply_plan_edit_steps(
                model=model, tokenizer=tokenizer, current_text=current_text,
                action=action, config=config, dinfer_engine=edit_dinfer_engine,
            )
            current_text = edit_record.pop("current_text")
            latest_plan_text = edit_record.get("latest_plan_raw_after_edit", latest_plan_text)
            action_record.update(edit_record)
            actions.append(action_record)
            continue

        action_record["error"] = "unknown_action"
        actions.append(action_record)
        stop_reason = "unknown_action"
        break

    full_output = current_text[len(prompt) :]
    full_output = render_internal_tokens(full_output, config.sampling)
    latest_plan = render_internal_tokens(latest_plan_text, config.sampling)
    final_answer = render_internal_tokens(final_answer_text, config.sampling)
    return RolloutResult(
        question=question,
        prompt=prompt,
        full_output=full_output,
        final_answer=final_answer,
        latest_plan=latest_plan,
        actions=actions,
        retrieval_used=retrieval_used,
        generated_tokens_model_only=generated_tokens,
        stop_reason=stop_reason,
    )


def _apply_plan_edit(**kwargs):
    steps = _apply_plan_edit_steps(**kwargs)
    response = None
    while True:
        try:
            request = steps.send(response)
        except StopIteration as done:
            return done.value
        response = _execute_request(request)


def _apply_plan_edit_steps(
    *,
    model: Any,
    tokenizer: Any,
    current_text: str,
    action: Action,
    config: RolloutConfig,
    dinfer_engine: Any = None,
) -> dict[str, Any]:
    plan_span = find_latest_plan_span(current_text, assistant_only=True)
    if plan_span is None:
        return {
            "current_text": current_text,
            "plan_edit_error": "no_plan",
        }

    plan_start, plan_end, plan_inner = plan_span
    edit = build_plan_edit(action.kind, action.content, plan_inner)
    if not edit.spans:
        return {
            "current_text": current_text,
            "plan_edit_error": "no_matching_span",
            "requested_markers": edit.requested_markers,
            "fallback_suffix": edit.fallback_suffix,
        }

    denoise_fn = denoise_plan_edit_dinfer if dinfer_engine is not None else denoise_plan_edit
    denoise_kwargs = {
        "tokenizer": tokenizer,
        "prefix_text": current_text[:plan_start],
        "plan_inner": plan_inner,
        "suffix_text": current_text[plan_end:],
        "edit": edit,
        "cfg": config.sampling,
    }
    if dinfer_engine is not None:
        new_plan_inner, replacements, info = yield from denoise_plan_edit_steps(
            dinfer_engine=dinfer_engine, **denoise_kwargs)
    else:
        new_plan_inner, replacements, info = denoise_fn(model=model, **denoise_kwargs)
    updated_text = replace_latest_plan_inner(current_text, new_plan_inner, assistant_only=True)
    return {
        "current_text": updated_text,
        "requested_markers": edit.requested_markers,
        "fallback_suffix": edit.fallback_suffix,
        "masked_spans": [span.__dict__ for span in edit.spans],
        "replacements": replacements,
        "plan_edit": info,
        "latest_plan_raw_after_edit": new_plan_inner,
        "latest_plan_after_edit": render_internal_tokens(new_plan_inner, config.sampling),
    }


def _truncate_by_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=False)


def normalize_search_query(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def config_to_dict(config: RolloutConfig) -> dict[str, Any]:
    return asdict(config)
