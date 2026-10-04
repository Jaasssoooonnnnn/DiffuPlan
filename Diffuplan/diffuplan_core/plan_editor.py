"""Editable-plan span parsing and mask request construction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from .actions import parse_marker_list


_Q_STEP_RE = re.compile(r"#Q_(\d+)\s*:")
_ROLE_RE = re.compile(r"<role>(SYSTEM|HUMAN|ASSISTANT)</role>")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    owner: str


@dataclass(frozen=True)
class PlanEdit:
    kind: str
    requested_markers: list[str]
    spans: list[Span]
    fallback_suffix: bool = False


def find_latest_plan_span(text: str, *, assistant_only: bool = False) -> Optional[tuple[int, int, str]]:
    stack: list[tuple[int, str]] = []
    latest: Optional[tuple[int, int]] = None
    for match in re.finditer(r"</?plan>", text or ""):
        if match.group(0) == "<plan>":
            stack.append((match.end(), _role_at(text or "", match.start())))
            continue
        if stack:
            start, role = stack.pop()
            if assistant_only and role not in {"", "ASSISTANT"}:
                continue
            latest = (start, match.start())
    if latest is None:
        return None
    start, end = latest
    return start, end, (text or "")[start:end]


def replace_latest_plan_inner(text: str, new_inner: str, *, assistant_only: bool = False) -> str:
    plan_span = find_latest_plan_span(text, assistant_only=assistant_only)
    if plan_span is None:
        return text
    start, end, _ = plan_span
    return text[:start] + new_inner + text[end:]


def _role_at(text: str, pos: int) -> str:
    role = ""
    for match in _ROLE_RE.finditer(text[:pos]):
        role = match.group(1)
    return role


def build_plan_edit(kind: str, action_content: str, plan_inner: str) -> PlanEdit:
    markers = parse_marker_list(action_content)
    if kind == "update":
        spans = _build_update_spans(markers, plan_inner)
        return PlanEdit(kind=kind, requested_markers=markers, spans=spans)
    if kind == "replan":
        spans, fallback = _build_replan_spans(markers, plan_inner)
        return PlanEdit(kind=kind, requested_markers=markers, spans=spans, fallback_suffix=fallback)
    raise ValueError(f"Unsupported plan edit kind: {kind}")


def _build_update_spans(markers: list[str], plan_inner: str) -> list[Span]:
    spans: list[Span] = []
    for marker in markers:
        if not re.fullmatch(r"#A_\d+", marker):
            continue
        idx = plan_inner.find(marker)
        if idx >= 0:
            spans.append(Span(start=idx, end=idx + len(marker), owner=marker))
    return _dedupe_spans(spans)


def _build_replan_spans(markers: list[str], plan_inner: str) -> tuple[list[Span], bool]:
    q_markers = [marker for marker in markers if re.fullmatch(r"#Q_\d+", marker)]
    step_spans = _find_q_step_spans(plan_inner)
    if not q_markers or not step_spans:
        return [], False

    if len(q_markers) == 1:
        start_step = q_markers[0]
        start_idx = next((span.start for span in step_spans if span.owner == start_step), None)
        if start_idx is None:
            return [], True
        return [Span(start=start_idx, end=len(plan_inner), owner=f"{start_step}:suffix")], True

    selected = [span for span in step_spans if span.owner in set(q_markers)]
    return selected, False


def _find_q_step_spans(plan_inner: str) -> list[Span]:
    matches = list(_Q_STEP_RE.finditer(plan_inner or ""))
    spans: list[Span] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(plan_inner)
        spans.append(Span(start=match.start(), end=end, owner=f"#Q_{match.group(1)}"))
    return spans


def _dedupe_spans(spans: list[Span]) -> list[Span]:
    seen: set[tuple[int, int, str]] = set()
    deduped: list[Span] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end, item.owner)):
        key = (span.start, span.end, span.owner)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    return deduped


def char_spans_to_token_owners(
    tokenizer: Any,
    text: str,
    spans: list[Span],
) -> tuple[list[int], list[Optional[str]]]:
    """Tokenize text and mark tokens overlapping requested character spans."""

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded.get("offset_mapping", []))
    owners: list[Optional[str]] = [None] * len(input_ids)
    for token_idx, (tok_start, tok_end) in enumerate(offsets):
        for span in spans:
            if tok_end > span.start and tok_start < span.end:
                owners[token_idx] = span.owner
                break
    return input_ids, owners


def update_spans_to_mask_slots(
    tokenizer: Any,
    text: str,
    spans: list[Span],
    *,
    mask_token_id: int,
) -> tuple[list[int], list[Optional[str]]]:
    """Compile each update marker span into one logical editable mask slot."""

    input_ids: list[int] = []
    owners: list[Optional[str]] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if span.start < cursor:
            continue
        prefix_ids = tokenizer.encode(text[cursor : span.start], add_special_tokens=False)
        input_ids.extend(prefix_ids)
        owners.extend([None] * len(prefix_ids))
        input_ids.append(int(mask_token_id))
        owners.append(span.owner)
        cursor = span.end

    suffix_ids = tokenizer.encode(text[cursor:], add_special_tokens=False)
    input_ids.extend(suffix_ids)
    owners.extend([None] * len(suffix_ids))
    return input_ids, owners


def apply_update_duplicate_replacements(plan_inner: str, replacements: dict[str, str]) -> str:
    """Replace unmasked duplicate #A_i markers with the denoised answer text."""

    updated = plan_inner
    for marker, replacement in replacements.items():
        replacement = (replacement or "").strip()
        if not marker.startswith("#A_") or not replacement:
            continue
        updated = updated.replace(marker, replacement)
    return updated
