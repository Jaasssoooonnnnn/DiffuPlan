"""Run editable-plan rollouts and collect action-token and edit-canvas traces.

The shared dInfer engine handles both modes. Action traces contain token IDs and
denoising commit steps; edit traces contain the canvas, targets, and ownership."""

from __future__ import annotations

import time
from typing import Any, Optional

from diffuplan_core.actions import ACTION_TAGS
from diffuplan_core.rollout import RolloutConfig, run_editable_plan_rollout
from diffuplan_core.search_client import SearchClient


# Markers an episode's plan must contain at least one of, for an update to ever be possible.
_ANSWER_PLACEHOLDER = "#A_"

# Parser status is diagnostic, not an RL loss filter. Failed model actions must
# receive their episode advantage whenever their token-space trace is available.
_MALFORMED_ACTION_ERRORS = {"no_complete_action", "unknown_action", "answer_direct_expected"}


def run_traced_episode(
    *,
    model: Any,
    tokenizer: Any,
    question: str,
    reference: Any,
    search_client: Optional[SearchClient],
    config: RolloutConfig,
    action_dinfer_engine: Any = None,
    edit_dinfer_engine: Any = None,
    input_index: int = 0,
    sample_index: int = 0,
    dataset_name: str = "",
    fail_on_error: bool = False,
) -> dict[str, Any]:
    """Run one rollout sample and return an episode dict with traces + reward fields."""

    t0 = time.time()
    try:
        result = run_editable_plan_rollout(
            model=model,
            tokenizer=tokenizer,
            question=question,
            search_client=search_client,
            config=config,
            action_dinfer_engine=action_dinfer_engine,
            edit_dinfer_engine=edit_dinfer_engine,
            collect_trace=True,
        )
    except Exception as exc:  # noqa: BLE001 - rollout must not kill the whole batch
        if fail_on_error:
            raise
        return _failed_episode(exc, question=question, reference=reference,
                               input_index=input_index, sample_index=sample_index,
                               dataset_name=dataset_name, elapsed_s=time.time() - t0)

    return _episode_from_result(result, question=question, reference=reference,
                                input_index=input_index, sample_index=sample_index,
                                dataset_name=dataset_name, elapsed_s=time.time() - t0)


def _episode_from_result(result, *, question, reference, input_index,
                         sample_index, dataset_name, elapsed_s):
    action_traces, edit_traces = _split_traces(result.actions)
    counts = _count_actions(result.actions)
    return {
        "input_index": int(input_index),
        "sample_index": int(sample_index),
        "dataset_name": dataset_name,
        "question": question,
        "reference": reference,
        "ground_truth": reference,
        "prediction": result.final_answer,
        "full_output": result.full_output,
        "latest_plan": result.latest_plan,
        "stop_reason": result.stop_reason,
        "retrieval_used": int(result.retrieval_used),
        "generated_tokens_model_only": int(result.generated_tokens_model_only),
        "response_length": len(result.full_output),
        "elapsed_s": elapsed_s,
        "actions": result.actions,
        # Flattened traces for trace_to_rows.py.
        "action_traces": action_traces,
        "edit_traces": edit_traces,
        # Aggregates used by the reward/format guard (cheap, computed once here).
        "num_update": counts["update"],
        "num_replan": counts["replan"],
        "num_search": counts["search"],
        "num_answer": counts["answer"],
        "plan_has_placeholder": _ANSWER_PLACEHOLDER in (result.latest_plan or ""),
    }


def _split_traces(actions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pull per-segment action traces and per-edit traces out of the action records."""

    action_traces: list[dict[str, Any]] = []
    edit_traces: list[dict[str, Any]] = []
    for record in actions or []:
        kind = record.get("kind")
        trace = record.get("trace")
        is_well_formed = (kind in ACTION_TAGS) and (record.get("error") not in _MALFORMED_ACTION_ERRORS)
        if trace and (trace.get("segment_token_ids") or trace.get("conditioning_token_ids")):
            action_traces.append(
                {
                    **trace,
                    "turn": int(record.get("turn", 0)),
                    "kind": kind,
                    "parse_valid": is_well_formed,
                    "parse_error": record.get("error"),
                    "prefix_text": trace.get("prefix_text", ""),
                    "segment_text": trace.get("segment_text", ""),
                    "segment_token_ids": list(trace.get("segment_token_ids", []) or []),
                    "segment_step_map": list(trace.get("segment_step_map", []) or []),
                }
            )
        if kind in {"update", "replan"}:
            plan_edit = record.get("plan_edit") or {}
            et = plan_edit.get("edit_trace")
            if et and et.get("final_plan_ids"):
                edit_traces.append(
                    {
                        "turn": int(record.get("turn", 0)),
                        "kind": kind,
                        "status": plan_edit.get("status", ""),
                        "expanded": bool(plan_edit.get("expanded", 0)),
                        "edit_events": list(plan_edit.get("edit_events", []) or []),
                        "edit_trace": et,
                    }
                )
    return action_traces, edit_traces


def _count_actions(actions: list[dict[str, Any]]) -> dict[str, int]:
    counts = {tag: 0 for tag in ACTION_TAGS}
    for record in actions or []:
        kind = record.get("kind")
        if kind in counts:
            counts[kind] += 1
    return counts


def _failed_episode(exc, *, question, reference, input_index, sample_index,
                    dataset_name, elapsed_s):
    return {
        "input_index": int(input_index),
        "sample_index": int(sample_index),
        "dataset_name": dataset_name,
        "question": question,
        "reference": reference,
        "ground_truth": reference,
        "prediction": "",
        "full_output": "",
        "latest_plan": "",
        "stop_reason": "exception",
        "retrieval_used": 0,
        "generated_tokens_model_only": 0,
        "response_length": 0,
        "elapsed_s": elapsed_s,
        "actions": [],
        "action_traces": [],
        "edit_traces": [],
        "num_update": 0,
        "num_replan": 0,
        "num_search": 0,
        "num_answer": 0,
        "plan_has_placeholder": False,
        "error": f"{type(exc).__name__}: {exc}",
        }


def run_traced_episodes(*, tasks, tokenizer, search_client, config, dinfer_engine,
                        batch_size, on_done=None):
    from diffuplan_core.rollout import run_editable_plan_rollouts
    begin = time.time()
    episodes = [None] * len(tasks)
    kwargs = [dict(model=None, tokenizer=tokenizer, question=item["question"],
                   search_client=search_client, config=config,
                   action_dinfer_engine=dinfer_engine, edit_dinfer_engine=dinfer_engine,
                   collect_trace=True) for item in tasks]

    def completed(index, result):
        info = dict(tasks[index], elapsed_s=time.time() - begin)
        if isinstance(result, Exception):
            episode = _failed_episode(result, **info)
        else:
            episode = _episode_from_result(result, **info)
        episodes[index] = episode
        if on_done:
            on_done(index, episode)

    run_editable_plan_rollouts(episode_kwargs=kwargs, batch_size=batch_size, on_done=completed)
    return episodes
