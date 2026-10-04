"""Reconstruct Action-Mode training rows from rollout traces.

Rows preserve the recorded response token IDs and commit steps and carry the
episode-level advantage."""

from __future__ import annotations

import random
from typing import Any, Optional


def build_action_rows(
    episode: dict[str, Any],
    *,
    reward: float,
    loss_weight: float,
    max_segments: int = 0,
    rng: Optional[random.Random] = None,
) -> list[dict[str, Any]]:
    """Build per-action-segment rows. Applies the segment cap (decision: unbiased subset)."""

    traces = list(episode.get("action_traces", []) or [])
    traces = [t for t in traces if (t.get("segment_token_ids") and t.get("segment_step_map"))
              or (t.get("trace_version") == 2 and any(t.get("action_loss_mask", [])))]
    traces = [t for t in traces if len(t["segment_token_ids"]) == len(t["segment_step_map"])]
    if not traces:
        return []

    selected = _select_segments(traces, max_segments=max_segments, rng=rng)

    rows: list[dict[str, Any]] = []
    for t in selected:
        rows.append(
            {
                "row_type": "action",
                **{key: t[key] for key in (
                    "trace_version", "prompt_token_ids", "conditioning_token_ids",
                    "conditioning_step_map", "action_loss_mask", "sampling_temperature",
                    "banned_token_ids", "parse_valid", "parse_error") if key in t},
                "prompt_text": t.get("prefix_text", ""),
                "response_token_ids": list(t["segment_token_ids"]),
                "step_map": list(t["segment_step_map"]),
                "kind": t.get("kind"),
                "turn": int(t.get("turn", 0)),
                "reward": float(reward),
                "loss_weight": float(loss_weight),
                "input_index": int(episode.get("input_index", 0)),
                "sample_index": int(episode.get("sample_index", 0)),
            }
        )
    return rows


def _select_segments(
    traces: list[dict[str, Any]],
    *,
    max_segments: int,
    rng: Optional[random.Random],
) -> list[dict[str, Any]]:
    """Keep first segment + terminal segment + random middle subset (unbiased under episode advantage)."""

    if max_segments is None or max_segments <= 0 or len(traces) <= max_segments:
        return traces

    rng = rng or random.Random()
    first_idx = 0
    last_idx = len(traces) - 1
    forced = {first_idx, last_idx}
    middle = [i for i in range(len(traces)) if i not in forced]
    n_middle = max(0, max_segments - len(forced))
    chosen_middle = set(rng.sample(middle, min(n_middle, len(middle)))) if n_middle > 0 else set()
    keep = sorted(forced | chosen_middle)
    return [traces[i] for i in keep]
