#!/usr/bin/env python
"""Build canonical DiffuPlan action and edit SFT records from teacher traces."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFUPLAN_ROOT = REPO_ROOT / "Diffuplan"
for item in (REPO_ROOT, DIFFUPLAN_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from diffuplan_core.prompting import build_answer_direct_observation, build_information_observation, build_prompt


DEFAULT_TOKENIZER = str(REPO_ROOT / "weights" / "base")
MASK_TOKEN_ID = 156895
EXPAND_TOKEN_ID = 156901
EXPAND_TOKEN_TEXT = "<|reserved_token_6|>"

PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
REPLAN_RE = re.compile(r"<replan>(.*?)</replan>", re.DOTALL | re.IGNORECASE)
INFO_RE = re.compile(r"<information>\n?(.*?)\n?</information>", re.DOTALL | re.IGNORECASE)
UPDATED_RE = re.compile(r"<updated_#Q_(\d+)>(.*?)</updated_#Q_\1>", re.DOTALL | re.IGNORECASE)
SUBANSWER_RE = re.compile(r"^\s*#A_(\d+)\s*:\s*(.+?)\s*$", re.MULTILINE)
Q_LINE_RE = re.compile(r"^\s*#Q_(\d+)\s*:\s*(.+?)\s*$", re.MULTILINE)
ANSWER_REF_RE = re.compile(r"#A_(\d+)")
PLAN_SPAN_RE = re.compile(r"</?plan>", re.IGNORECASE)
HUMAN_ROLE_MARKER = "<|role_end|><role>HUMAN</role>"
ASSISTANT_ROLE_MARKER = "<|role_end|><role>ASSISTANT</role>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="accepted_canonical_clean.jsonl")
    parser.add_argument(
        "--full-trace-input",
        action="append",
        default=[],
        help="teacher_trace_text JSONL to convert fully into action/edit rows.",
    )
    parser.add_argument(
        "--answer-direct-trace-input",
        action="append",
        default=[],
        help="teacher_trace_text JSONL whose last answer-direct action should be exported only.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--update-states", type=int, default=12)
    parser.add_argument("--replan-states", type=int, default=24)
    parser.add_argument("--max-target-tokens", type=int, default=256)
    parser.add_argument("--dreamon-mask-prob-low", type=float, default=0.25)
    parser.add_argument("--dreamon-mask-prob-high", type=float, default=1.0)
    parser.add_argument("--dreamon-merge-prob", type=float, default=0.55)
    parser.add_argument("--max-traces", type=int, default=None)
    args = parser.parse_args()
    if not args.input and not args.full_trace_input and not args.answer_direct_trace_input:
        parser.error("provide --input, --full-trace-input, or --answer-direct-trace-input")
    return args


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SftRecordWriter(output_dir=output_dir, seed=int(args.seed) + 1009)
    stats = Counter()

    input_paths = {
        "canonical": [str(Path(args.input))] if args.input else [],
        "full_trace": [str(Path(item)) for item in args.full_trace_input],
        "answer_direct_trace": [str(Path(item)) for item in args.answer_direct_trace_input],
    }

    canonical_inputs = [Path(args.input)] if args.input else []
    for input_path in canonical_inputs:
        for line_no, payload in iter_jsonl(input_path, max_rows=args.max_traces):
            stats["canonical_rows"] += 1
            stats["loaded_traces"] += 1
            state = build_trace_state(
                payload=payload,
                line_no=line_no,
                tokenizer=tokenizer,
                rng=rng,
                update_states=int(args.update_states),
                replan_states=int(args.replan_states),
                max_target_tokens=int(args.max_target_tokens),
                mask_prob_low=float(args.dreamon_mask_prob_low),
                mask_prob_high=float(args.dreamon_mask_prob_high),
                merge_prob=float(args.dreamon_merge_prob),
            )
            writer.write_actions(state["actions"])
            writer.write_edits(state["edits"])
            stats.update(state["stats"])

    for input_path in [Path(item) for item in args.full_trace_input]:
        for line_no, row in iter_jsonl(input_path, max_rows=args.max_traces):
            stats["full_trace_rows"] += 1
            stats["loaded_traces"] += 1
            payload = teacher_trace_text_payload(
                row=row,
                line_no=line_no,
                input_path=input_path,
                mode="full_trace",
            )
            state = build_trace_state(
                payload=payload,
                line_no=line_no,
                tokenizer=tokenizer,
                rng=rng,
                update_states=int(args.update_states),
                replan_states=int(args.replan_states),
                max_target_tokens=int(args.max_target_tokens),
                mask_prob_low=float(args.dreamon_mask_prob_low),
                mask_prob_high=float(args.dreamon_mask_prob_high),
                merge_prob=float(args.dreamon_merge_prob),
            )
            writer.write_actions(state["actions"])
            writer.write_edits(state["edits"])
            stats.update(state["stats"])

    for input_path in [Path(item) for item in args.answer_direct_trace_input]:
        for line_no, row in iter_jsonl(input_path, max_rows=args.max_traces):
            stats["answer_direct_trace_rows"] += 1
            stats["loaded_traces"] += 1
            record, row_stats = build_answer_direct_record_from_teacher_text(
                row=row,
                line_no=line_no,
                input_path=input_path,
                tokenizer=tokenizer,
                rng=rng,
                update_states=int(args.update_states),
                replan_states=int(args.replan_states),
                max_target_tokens=int(args.max_target_tokens),
                mask_prob_low=float(args.dreamon_mask_prob_low),
                mask_prob_high=float(args.dreamon_mask_prob_high),
                merge_prob=float(args.dreamon_merge_prob),
            )
            writer.write_action(record)
            stats.update(row_stats)

    writer.finish()

    summary = {
        "schema_version": "diffuplan_sft_canonical",
        "inputs": input_paths,
        "tokenizer_path": str(args.tokenizer_path),
        "mask_token_id": MASK_TOKEN_ID,
        "expand_token_id": EXPAND_TOKEN_ID,
        "expand_token_text": EXPAND_TOKEN_TEXT,
        "loaded_traces": stats["loaded_traces"],
        "canonical_rows": stats["canonical_rows"],
        "full_trace_rows": stats["full_trace_rows"],
        "answer_direct_trace_rows": stats["answer_direct_trace_rows"],
        "action_records": writer.action_records,
        "edit_records": writer.edit_records,
        "mixed_records": writer.mixed_records,
        "action_type_counts": writer.action_type_counts,
        "edit_type_counts": writer.edit_type_counts,
        "edit_state_counts": writer.edit_state_counts,
        "stats": stats,
        "audit": {"action_bad": writer.action_bad, "edit_bad": writer.edit_bad},
        "outputs": {
            "action_sft": str(output_dir / "action_sft.jsonl"),
            "edit_sft": str(output_dir / "edit_sft.jsonl"),
            "mixed_sft": str(output_dir / "mixed_sft.jsonl"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
    return 0


class SftRecordWriter:
    def __init__(self, *, output_dir: Path, seed: int, bucket_count: int = 64) -> None:
        self.output_dir = output_dir
        self.action_path = output_dir / "action_sft.jsonl"
        self.edit_path = output_dir / "edit_sft.jsonl"
        self.mixed_path = output_dir / "mixed_sft.jsonl"
        self.rng = random.Random(seed)
        self.bucket_paths = [output_dir / f".mixed_sft.bucket_{idx:03d}.tmp" for idx in range(int(bucket_count))]
        self.bucket_files = [path.open("w", encoding="utf-8") for path in self.bucket_paths]
        self.action_file = self.action_path.open("w", encoding="utf-8")
        self.edit_file = self.edit_path.open("w", encoding="utf-8")
        self.action_records = 0
        self.edit_records = 0
        self.mixed_records = 0
        self.action_type_counts: Counter = Counter()
        self.edit_type_counts: Counter = Counter()
        self.edit_state_counts: Counter = Counter()
        self.action_bad: Counter = Counter()
        self.edit_bad: Counter = Counter()
        self._finished = False

    def write_actions(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self.write_action(record)

    def write_edits(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self.write_edit(record)

    def write_action(self, record: dict[str, Any]) -> None:
        self.action_records += 1
        self.mixed_records += 1
        self.action_type_counts[record.get("action_type")] += 1
        self._audit_action(record)
        line = json.dumps(flatten_record(record), ensure_ascii=False) + "\n"
        self.action_file.write(line)
        self._write_mixed_line(line)

    def write_edit(self, record: dict[str, Any]) -> None:
        self.edit_records += 1
        self.mixed_records += 1
        self.edit_type_counts[record.get("edit_type")] += 1
        self.edit_state_counts[record.get("state_kind")] += 1
        self._audit_edit(record)
        line = json.dumps(flatten_record(record), ensure_ascii=False) + "\n"
        self.edit_file.write(line)
        self._write_mixed_line(line)

    def finish(self) -> None:
        if self._finished:
            return
        self.action_file.close()
        self.edit_file.close()
        for bucket_file in self.bucket_files:
            bucket_file.close()
        order = list(range(len(self.bucket_paths)))
        self.rng.shuffle(order)
        with self.mixed_path.open("w", encoding="utf-8") as out:
            for idx in order:
                path = self.bucket_paths[idx]
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        out.write(line)
                path.unlink(missing_ok=True)
        self._finished = True

    def _write_mixed_line(self, line: str) -> None:
        bucket = self.rng.randrange(len(self.bucket_files))
        self.bucket_files[bucket].write(line)

    def _audit_action(self, record: dict[str, Any]) -> None:
        target = record.get("target_text", "")
        if re.search(r"^\s*#A_\d+\s*:", strip_update_replan_content(target), re.MULTILINE):
            self.action_bad["raw_answer_helper_in_action"] += 1
        if UPDATED_RE.search(target):
            self.action_bad["updated_q_block_in_action"] += 1
        if "<mask>" in target or "<expand>" in target or EXPAND_TOKEN_TEXT in target:
            self.action_bad["internal_token_in_action"] += 1
        if not has_terminal_action(target):
            self.action_bad["missing_terminal_action"] += 1

    def _audit_edit(self, record: dict[str, Any]) -> None:
        if not any(int(item) for item in record.get("loss_mask", [])):
            self.edit_bad["no_active_loss"] += 1
        if len(record.get("input_token_ids", [])) != len(record.get("label_token_ids", [])):
            self.edit_bad["length_mismatch"] += 1


def iter_jsonl(path: Path, *, max_rows: int | None = None) -> Iterable[tuple[int, dict[str, Any]]]:
    loaded = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            if max_rows is not None and loaded >= max_rows:
                break
            loaded += 1
            yield line_no, json.loads(line)


def teacher_trace_text_payload(
    *,
    row: dict[str, Any],
    line_no: int,
    input_path: Path,
    mode: str,
    trace_text: str | None = None,
) -> dict[str, Any]:
    if not str(row.get("schema_version", "")).startswith("teacher_trace_text"):
        raise ValueError(f"{input_path}:{line_no} is not a teacher_trace_text record")
    source_id = str(row.get("source_trace_id") or f"{input_path.stem}:line_{line_no:06d}")
    return {
        "schema_version": "plan_trace_from_teacher_trace_text",
        "sample_id": source_id,
        "data_source": row.get("data_source"),
        "question": row.get("question") or "",
        "gold_answers": row.get("gold_answers") or [],
        "prediction": row.get("prediction"),
        "scores": row.get("scores") or {},
        "accepted": row.get("accepted"),
        "events": trace_text_to_events(str(row.get("trace_text") if trace_text is None else trace_text)),
        "metadata": teacher_trace_metadata(row, line_no=line_no, input_path=input_path, mode=mode),
    }


def teacher_trace_metadata(row: dict[str, Any], *, line_no: int, input_path: Path, mode: str) -> dict[str, Any]:
    metadata = {
        "trace_input_mode": mode,
        "trace_input_path": str(input_path),
        "trace_input_line": line_no,
    }
    for key in (
        "schema_version",
        "source_trace_id",
        "source_line",
        "data_source",
        "source_dataset",
        "source_id",
        "source_file",
        "component",
        "accepted",
        "skip_reason",
        "search_budget",
        "retrieval_calls",
        "answer_direct_used",
        "answer_direct_reason",
        "prediction",
        "scores",
        "gold_answers",
    ):
        if key in row:
            metadata[key] = row.get(key)
    return metadata


def trace_text_to_events(trace_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cursor = 0
    for match in INFO_RE.finditer(trace_text or ""):
        assistant_text = strip_role_markers(trace_text[cursor:match.start()]).strip()
        if assistant_text:
            events.append({"kind": "model", "stage": "assistant_turn", "content": assistant_text})
        events.append({"kind": "observation", "stage": "search", "observation_text": str(match.group(1) or "").strip()})
        cursor = match.end()
    tail = strip_role_markers((trace_text or "")[cursor:]).strip()
    if tail:
        events.append({"kind": "model", "stage": "assistant_turn", "content": tail})
    return events


def build_answer_direct_record_from_teacher_text(
    *,
    row: dict[str, Any],
    line_no: int,
    input_path: Path,
    tokenizer: Any,
    rng: random.Random,
    update_states: int,
    replan_states: int,
    max_target_tokens: int,
    mask_prob_low: float,
    mask_prob_high: float,
    merge_prob: float,
) -> tuple[dict[str, Any], Counter]:
    del update_states, replan_states
    pre_budget_text, budget_text, final_segment = split_answer_direct_trace_text(str(row.get("trace_text") or ""))
    payload = teacher_trace_text_payload(
        row=row,
        line_no=line_no,
        input_path=input_path,
        mode="answer_direct_prefix",
        trace_text=pre_budget_text,
    )
    payload["events"].append({"kind": "observation", "stage": "answer_direct", "observation_text": budget_text})
    state = build_trace_state(
        payload=payload,
        line_no=line_no,
        tokenizer=tokenizer,
        rng=rng,
        update_states=1,
        replan_states=1,
        max_target_tokens=max_target_tokens,
        mask_prob_low=mask_prob_low,
        mask_prob_high=mask_prob_high,
        merge_prob=merge_prob,
    )
    target_text, target_metadata = clean_answer_direct_target(final_segment)
    metadata = merge_metadata(
        teacher_trace_metadata(row, line_no=line_no, input_path=input_path, mode="answer_direct_final"),
        {
            "target_scope": "answer_direct_final_segment",
            "answer_direct_observation_text": budget_text,
            "answer_direct_prompt_uses_controller_observation": True,
            **target_metadata,
        },
    )
    source_id = str(row.get("source_trace_id") or f"{input_path.stem}:line_{line_no:06d}")
    record = make_action_record(
        source_id,
        line_no,
        "answer",
        str(row.get("question") or ""),
        str(state["current_text"]),
        target_text,
        metadata,
    )
    stats = prefixed_counter(state["stats"], "answer_direct_prefix_")
    stats["answer_direct_action_records"] += 1
    if target_text.lstrip().lower().startswith("<answer>"):
        stats["answer_direct_answer_only_targets"] += 1
    else:
        stats["answer_direct_think_answer_targets"] += 1
    return record, stats


def split_answer_direct_trace_text(trace_text: str) -> tuple[str, str, str]:
    assistant_pos = trace_text.rfind(ASSISTANT_ROLE_MARKER)
    if assistant_pos < 0:
        raise ValueError("answer-direct trace is missing final assistant role marker")
    prefix_with_budget = trace_text[:assistant_pos]
    final_segment = trace_text[assistant_pos + len(ASSISTANT_ROLE_MARKER):].strip()
    human_pos = prefix_with_budget.rfind(HUMAN_ROLE_MARKER)
    if human_pos < 0:
        raise ValueError("answer-direct trace is missing budget human role marker")
    pre_budget_text = prefix_with_budget[:human_pos].strip()
    budget_text = prefix_with_budget[human_pos + len(HUMAN_ROLE_MARKER):].strip()
    if not final_segment:
        raise ValueError("answer-direct trace has an empty final assistant segment")
    return pre_budget_text, budget_text, final_segment


def clean_answer_direct_target(final_segment: str) -> tuple[str, dict[str, Any]]:
    artifact_counts = count_teacher_only_artifacts(final_segment)
    cleaned = strip_teacher_only_artifacts(strip_role_markers(final_segment)).strip()
    answer_match = ANSWER_RE.search(cleaned)
    if answer_match is None:
        raise ValueError("answer-direct final segment is missing an answer block")
    answer_text = str(answer_match.group(1) or "").strip()
    if not answer_text:
        raise ValueError("answer-direct final segment has an empty answer block")
    think = extract_think_before(cleaned, before_pos=answer_match.start())
    target = "\n".join([block("think", think), block("answer", answer_text)]) if think else block("answer", answer_text)
    if not is_answer_direct_target(target):
        raise ValueError("answer-direct target is not answer-only or think+answer")
    remaining_artifacts = count_teacher_only_artifacts(target)
    if remaining_artifacts:
        raise ValueError(f"answer-direct target still contains teacher-only artifacts: {dict(remaining_artifacts)}")
    return target, {
        "removed_raw_subanswer_lines": artifact_counts.get("raw_subanswer_lines", 0),
        "removed_updated_q_blocks": artifact_counts.get("updated_q_blocks", 0),
        "final_target_has_think": bool(think),
    }


def build_trace_records(
    *,
    payload: dict[str, Any],
    line_no: int,
    tokenizer: Any,
    rng: random.Random,
    update_states: int,
    replan_states: int,
    max_target_tokens: int,
    mask_prob_low: float,
    mask_prob_high: float,
    merge_prob: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    state = build_trace_state(
        payload=payload,
        line_no=line_no,
        tokenizer=tokenizer,
        rng=rng,
        update_states=update_states,
        replan_states=replan_states,
        max_target_tokens=max_target_tokens,
        mask_prob_low=mask_prob_low,
        mask_prob_high=mask_prob_high,
        merge_prob=merge_prob,
    )
    return state["actions"], state["edits"], state["stats"]


def build_trace_state(
    *,
    payload: dict[str, Any],
    line_no: int,
    tokenizer: Any,
    rng: random.Random,
    update_states: int,
    replan_states: int,
    max_target_tokens: int,
    mask_prob_low: float,
    mask_prob_high: float,
    merge_prob: float,
) -> dict[str, Any]:
    source_id = str(payload.get("sample_id") or "")
    question = str(payload.get("question") or "")
    current_text = build_prompt(question)
    latest_plan = ""
    answer_bank: dict[str, str] = {}
    actions: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    stats = Counter()
    base_metadata = dict(payload.get("metadata") or {})

    for event in payload.get("events") or []:
        kind = event.get("kind")
        stage = event.get("stage")
        if kind == "observation" and stage == "search":
            current_text += build_information_observation(str(event.get("observation_text") or ""))
            stats["observations"] += 1
            continue
        if kind == "observation" and stage == "answer_direct":
            # Search budget exhausted: inject the SAME controller observation the RL rollout uses
            # (build_answer_direct_observation), so the following answer action is trained under the
            # exact "action budget is exhausted -> answer now" context the policy meets at RL time.
            current_text += build_answer_direct_observation()
            stats["answer_direct_observations"] += 1
            continue
        if kind != "model" or stage != "assistant_turn":
            continue

        content = str(event.get("content") or "").strip()
        segments, seg_edits, latest_plan, current_text, answer_updates, seg_stats = canonicalize_assistant_turn(
            content=content,
            current_text=current_text,
            latest_plan=latest_plan,
            answer_bank=answer_bank,
            source_id=source_id,
            line_no=line_no,
            question=question,
            base_metadata=base_metadata,
            tokenizer=tokenizer,
            rng=rng,
            update_states=update_states,
            replan_states=replan_states,
            max_target_tokens=max_target_tokens,
            mask_prob_low=mask_prob_low,
            mask_prob_high=mask_prob_high,
            merge_prob=merge_prob,
        )
        answer_bank.update(answer_updates)
        actions.extend(segments)
        edits.extend(seg_edits)
        stats.update(seg_stats)

    return {
        "actions": actions,
        "edits": edits,
        "stats": stats,
        "current_text": current_text,
        "latest_plan": latest_plan,
        "answer_bank": answer_bank,
    }


def canonicalize_assistant_turn(
    *,
    content: str,
    current_text: str,
    latest_plan: str,
    answer_bank: dict[str, str],
    source_id: str,
    line_no: int,
    question: str,
    base_metadata: dict[str, Any],
    tokenizer: Any,
    rng: random.Random,
    update_states: int,
    replan_states: int,
    max_target_tokens: int,
    mask_prob_low: float,
    mask_prob_high: float,
    merge_prob: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str, dict[str, str], Counter]:
    actions: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    stats = Counter()
    answer_updates = extract_subanswers(content)

    plan_match = PLAN_RE.search(content)
    search_match = SEARCH_RE.search(content)
    answer_match = ANSWER_RE.search(content)
    replan_match = REPLAN_RE.search(content)
    updated_matches = list(UPDATED_RE.finditer(content))

    if plan_match and not latest_plan:
        initial_plan = clean_plan_inner(plan_match.group(1))
        if not initial_plan:
            stats["skip_empty_initial_plan"] += 1
            return actions, edits, latest_plan, current_text, answer_updates, stats
        if not search_match:
            stats["skip_initial_without_search"] += 1
            return actions, edits, latest_plan, current_text, answer_updates, stats
        think = extract_think_before(content, before_pos=search_match.start())
        if not think:
            stats["skip_initial_missing_think"] += 1
            return actions, edits, latest_plan, current_text, answer_updates, stats
        latest_plan = initial_plan
        segment = "\n".join([
            block("plan", latest_plan),
            block("think", think),
            block("search", normalize_space(search_match.group(1))),
        ])
        actions.append(make_action_record(source_id, line_no, "search", question, current_text, segment, merge_metadata(base_metadata, {"initial_plan": True})))
        current_text += segment
        stats["action_search"] += 1
        return actions, edits, latest_plan, current_text, answer_updates, stats

    answer_bank_for_turn = dict(answer_bank)
    answer_bank_for_turn.update(answer_updates)

    if replan_match:
        markers, target_suffix = replan_markers_and_target(replan_match.group(1))
        if markers and target_suffix:
            think = extract_think_before(content, before_pos=replan_match.start())
            if not think:
                stats["skip_replan_missing_think"] += 1
            else:
                segment = "\n".join([block("think", think), block("replan", ",".join(markers))])
                actions.append(make_action_record(source_id, line_no, "replan", question, current_text, segment, merge_metadata(base_metadata, {"markers": markers})))
                current_text += segment
                stats["action_replan"] += 1
                edit_records, latest_plan, current_text, edit_stats = build_replan_edit_records(
                    source_id=source_id,
                    line_no=line_no,
                    question=question,
                    base_metadata=base_metadata,
                    current_text=current_text,
                    latest_plan=latest_plan,
                    markers=markers,
                    target_suffix=target_suffix,
                    tokenizer=tokenizer,
                    rng=rng,
                    states_per_example=replan_states,
                    max_target_tokens=max_target_tokens,
                    mask_prob_low=mask_prob_low,
                    mask_prob_high=mask_prob_high,
                    merge_prob=merge_prob,
                )
                edits.extend(edit_records)
                stats.update(edit_stats)
        else:
            stats["skip_replan_parse"] += 1

    update_plan_text = apply_updated_blocks(latest_plan, updated_matches)
    update_markers = infer_update_markers(latest_plan, updated_matches, answer_bank_for_turn)
    update_replacements: dict[str, str] = {}
    for marker in update_markers:
        answer = answer_bank_for_turn.get(marker, "").strip()
        if not answer:
            stats["skip_update_missing_answer"] += 1
            continue
        if marker not in latest_plan:
            stats["skip_update_marker_not_in_plan"] += 1
            continue
        think = extract_think_before(content, before_pos=update_think_anchor(content, marker, updated_matches))
        if not think:
            stats["skip_update_missing_think"] += 1
            continue
        update_replacements[marker] = answer
        segment = "\n".join([block("think", think), block("update", marker)])
        actions.append(make_action_record(source_id, line_no, "update", question, current_text, segment, merge_metadata(base_metadata, {"marker": marker})))
        current_text += segment
        stats["action_update"] += 1
        edit_records, latest_plan, current_text, edit_stats = build_update_edit_records(
            source_id=source_id,
            line_no=line_no,
            question=question,
            base_metadata=base_metadata,
            current_text=current_text,
            latest_plan=latest_plan,
            marker=marker,
            answer=answer,
            tokenizer=tokenizer,
            rng=rng,
            states_per_example=update_states,
            max_target_tokens=max_target_tokens,
            mask_prob_low=mask_prob_low,
            mask_prob_high=mask_prob_high,
            merge_prob=merge_prob,
        )
        edits.extend(edit_records)
        stats.update(edit_stats)

    if update_replacements and update_plan_text:
        latest_plan = apply_update_duplicate_replacements(update_plan_text, update_replacements)
        current_text = replace_latest_plan_inner(current_text, latest_plan)

    terminal_match = search_match or answer_match
    if terminal_match:
        if search_match:
            query = normalize_space(search_match.group(1))
            think = extract_think_before(content, before_pos=search_match.start())
            if not think:
                stats["skip_search_missing_think"] += 1
            else:
                segment = "\n".join([block("think", think), block("search", query)])
                actions.append(make_action_record(source_id, line_no, "search", question, current_text, segment, merge_metadata(base_metadata, {})))
                current_text += segment
                stats["action_search"] += 1
        elif answer_match:
            answer = str(answer_match.group(1) or "").strip()
            if answer:
                think = extract_think_before(content, before_pos=answer_match.start())
                if not think:
                    stats["skip_answer_missing_think"] += 1
                else:
                    segment = "\n".join([block("think", think), block("answer", answer)])
                    actions.append(make_action_record(source_id, line_no, "answer", question, current_text, segment, merge_metadata(base_metadata, {})))
                    current_text += segment
                    stats["action_answer"] += 1

    if not actions:
        stats["skip_turn_no_student_action"] += 1
    return actions, edits, latest_plan, current_text, answer_updates, stats


def build_update_edit_records(
    *,
    source_id: str,
    line_no: int,
    question: str,
    base_metadata: dict[str, Any],
    current_text: str,
    latest_plan: str,
    marker: str,
    answer: str,
    tokenizer: Any,
    rng: random.Random,
    states_per_example: int,
    max_target_tokens: int,
    mask_prob_low: float,
    mask_prob_high: float,
    merge_prob: float,
) -> tuple[list[dict[str, Any]], str, str, Counter]:
    stats = Counter()
    target_ids = tokenize(tokenizer, answer)
    if not valid_target_ids(target_ids, max_target_tokens):
        stats["skip_update_bad_target_len"] += 1
        return [], latest_plan, current_text, stats
    plan_span = find_latest_plan_span(current_text)
    if plan_span is None:
        stats["skip_update_no_plan_span"] += 1
        return [], latest_plan, current_text, stats
    plan_start, plan_end, plan_inner = plan_span
    span_rel = plan_inner.find(marker)
    if span_rel < 0:
        stats["skip_update_marker_absent"] += 1
        return [], latest_plan, current_text, stats
    abs_start = plan_start + span_rel
    abs_end = abs_start + len(marker)
    states = build_dreamon_states_for_canvas(
        tokenizer=tokenizer,
        full_text=current_text,
        span_start=abs_start,
        span_end=abs_end,
        target_ids=target_ids,
        states_per_example=states_per_example,
        rng=rng,
        seed_mask_counts=[1, len(target_ids)],
        mask_prob_low=mask_prob_low,
        mask_prob_high=mask_prob_high,
        merge_prob=merge_prob,
    )
    new_plan = apply_update_duplicate_replacements(latest_plan, {marker: answer})
    new_text = replace_latest_plan_inner(current_text, new_plan)
    records = [
        make_edit_record(
            source_id=source_id,
            line_no=line_no,
            edit_type="update",
            state_kind="answer_slot",
            question=question,
            action_text=block("update", marker),
            target_text=answer,
            target_ids=target_ids,
            state=state,
            metadata=merge_metadata(base_metadata, {"marker": marker, "variant_index": idx}),
        )
        for idx, state in enumerate(states)
    ]
    stats["edit_update"] += len(records)
    return records, new_plan, new_text, stats


def build_replan_edit_records(
    *,
    source_id: str,
    line_no: int,
    question: str,
    base_metadata: dict[str, Any],
    current_text: str,
    latest_plan: str,
    markers: list[str],
    target_suffix: str,
    tokenizer: Any,
    rng: random.Random,
    states_per_example: int,
    max_target_tokens: int,
    mask_prob_low: float,
    mask_prob_high: float,
    merge_prob: float,
) -> tuple[list[dict[str, Any]], str, str, Counter]:
    stats = Counter()
    if not markers:
        stats["skip_replan_no_markers"] += 1
        return [], latest_plan, current_text, stats
    target_ids = tokenize(tokenizer, target_suffix)
    if not valid_target_ids(target_ids, max_target_tokens):
        stats["skip_replan_bad_target_len"] += 1
        return [], latest_plan, current_text, stats
    plan_span = find_latest_plan_span(current_text)
    if plan_span is None:
        stats["skip_replan_no_plan_span"] += 1
        return [], latest_plan, current_text, stats
    plan_start, _, plan_inner = plan_span
    span_rel = find_replan_span_start(plan_inner, markers[0])
    if span_rel < 0:
        stats["skip_replan_span_absent"] += 1
        return [], latest_plan, current_text, stats
    span_rel = move_past_leading_newlines(plan_inner, span_rel)
    span_end_rel = trim_trailing_newlines(plan_inner, len(plan_inner), min_start=span_rel)
    abs_start = plan_start + span_rel
    abs_end = plan_start + span_end_rel
    states = build_dreamon_states_for_canvas(
        tokenizer=tokenizer,
        full_text=current_text,
        span_start=abs_start,
        span_end=abs_end,
        target_ids=target_ids,
        states_per_example=states_per_example,
        rng=rng,
        seed_mask_counts=[1, len(target_ids)],
        mask_prob_low=mask_prob_low,
        mask_prob_high=mask_prob_high,
        merge_prob=merge_prob,
    )
    new_plan = plan_inner[:span_rel].rstrip() + ("\n" if span_rel > 0 else "") + target_suffix.strip()
    new_text = replace_latest_plan_inner(current_text, new_plan)
    action_text = block("replan", ",".join(markers))
    records = [
        make_edit_record(
            source_id=source_id,
            line_no=line_no,
            edit_type="replan",
            state_kind="plan_suffix",
            question=question,
            action_text=action_text,
            target_text=target_suffix,
            target_ids=target_ids,
            state=state,
            metadata=merge_metadata(base_metadata, {"markers": markers, "variant_index": idx}),
        )
        for idx, state in enumerate(states)
    ]
    stats["edit_replan"] += len(records)
    return records, new_plan, new_text, stats


def make_action_record(source_id: str, line_no: int, action_type: str, question: str, prompt_text: str, target_text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "diffuplan_sft_canonical",
        "record_type": "action",
        "source_trace_id": source_id,
        "source_line": line_no,
        "action_type": action_type,
        "question": question,
        "prompt_text": prompt_text,
        "target_text": target_text,
        "context": {"question": question, "uses_full_rollout_prefix": True},
        "metadata": metadata,
    }


def make_edit_record(
    *,
    source_id: str,
    line_no: int,
    edit_type: str,
    state_kind: str,
    question: str,
    action_text: str,
    target_text: str,
    target_ids: list[int],
    state: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    expand_labels = sum(1 for label, loss in zip(state["label_token_ids"], state["loss_mask"]) if loss and label == EXPAND_TOKEN_ID)
    return {
        "schema_version": "diffuplan_sft_canonical",
        "record_type": "edit",
        "source_trace_id": source_id,
        "source_line": line_no,
        "edit_type": edit_type,
        "state_kind": state_kind,
        "question": question,
        "action_text": action_text,
        "target_text": target_text,
        "target_token_ids": target_ids,
        "mask_token_id": MASK_TOKEN_ID,
        "expand_token_id": EXPAND_TOKEN_ID,
        "input_token_ids": state["input_token_ids"],
        "label_token_ids": state["label_token_ids"],
        "loss_mask": state["loss_mask"],
        "plan_input_rendered": state["input_rendered"],
        "active_mask_count": int(sum(state["loss_mask"])),
        "expand_label_count": int(expand_labels),
        "segments": state["segments"],
        "metadata": metadata,
    }


def build_dreamon_states_for_canvas(
    *,
    tokenizer: Any,
    full_text: str,
    span_start: int,
    span_end: int,
    target_ids: list[int],
    states_per_example: int,
    rng: random.Random,
    seed_mask_counts: list[int],
    mask_prob_low: float,
    mask_prob_high: float,
    merge_prob: float,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = set()
    prefix_ids = tokenize(tokenizer, full_text[:span_start])
    suffix_ids = tokenize(tokenizer, full_text[span_end:])

    def add(groups: list[tuple[str, list[int]]]) -> None:
        if len(states) >= max(1, states_per_example):
            return
        if not any(kind == "mask" for kind, _ in groups):
            return
        state = build_dreamon_state_for_canvas(
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            groups=groups,
        )
        key = (tuple(state["input_token_ids"]), tuple(state["label_token_ids"]), tuple(state["loss_mask"]))
        if key in seen:
            return
        seen.add(key)
        states.append(state)

    target_len = len(target_ids)
    for mask_count in unique_ints([1, target_len] + seed_mask_counts):
        if 1 <= mask_count <= target_len:
            add([("mask", segment) for segment in even_partition_ids(target_ids, mask_count)])
    if target_len > 1:
        for pos in unique_ints([0, target_len - 1, target_len // 2]):
            add([("mask", [tok]) if idx == pos else ("resolved", [tok]) for idx, tok in enumerate(target_ids)])
    attempts = 0
    while len(states) < max(1, states_per_example) and attempts < max(64, states_per_example * 64):
        attempts += 1
        add(sample_dreamon_groups(target_ids, rng, mask_prob_low, mask_prob_high, merge_prob))
    return states


def build_dreamon_state_for_canvas(
    *,
    prefix_ids: list[int],
    suffix_ids: list[int],
    groups: list[tuple[str, list[int]]],
) -> dict[str, Any]:
    pieces: list[int] = []
    labels: list[int] = []
    loss_mask: list[int] = []
    segments: list[dict[str, Any]] = []
    for idx, (kind, token_ids) in enumerate(groups):
        if kind == "mask":
            label = EXPAND_TOKEN_ID if len(token_ids) > 1 else token_ids[0]
            pieces.append(MASK_TOKEN_ID)
            labels.append(label)
            loss_mask.append(1)
            segments.append({"segment_index": idx, "state": "mask", "target_token_ids": token_ids, "label_token_id": label})
        else:
            for token_id in token_ids:
                pieces.append(token_id)
                labels.append(token_id)
                loss_mask.append(0)
            segments.append({"segment_index": idx, "state": "resolved", "target_token_ids": token_ids, "label_token_id": None})
    input_ids = prefix_ids + pieces + suffix_ids
    label_ids = prefix_ids + labels + suffix_ids
    full_loss_mask = [0] * len(prefix_ids) + loss_mask + [0] * len(suffix_ids)
    return {
        "input_token_ids": input_ids,
        "label_token_ids": label_ids,
        "loss_mask": full_loss_mask,
        "input_rendered": "",
        "segments": segments,
    }


def extract_subanswers(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in SUBANSWER_RE.finditer(text or ""):
        out[f"#A_{match.group(1)}"] = match.group(2).strip()
    return out


def infer_update_markers(plan: str, updated_matches: list[re.Match], answer_bank: dict[str, str]) -> list[str]:
    markers: list[str] = []
    steps = parse_plan_steps(plan)
    for match in updated_matches:
        step_id = int(match.group(1))
        before = steps.get(step_id, "")
        after = match.group(2)
        for ref in ANSWER_REF_RE.findall(before):
            marker = f"#A_{ref}"
            answer = answer_bank.get(marker, "")
            if not answer:
                continue
            if marker not in after or (answer and answer in after):
                if marker not in markers:
                    markers.append(marker)
    # Some teacher traces write raw helper lines such as "#A_1: Paris" without
    # also emitting <updated_#Q_j> blocks. In the student runtime, that still
    # corresponds to an explicit <update> handoff whenever the marker appears in
    # the current editable plan.
    for ref in sorted(set(ANSWER_REF_RE.findall(plan or "")), key=int):
        marker = f"#A_{ref}"
        if marker in answer_bank and marker not in markers:
            markers.append(marker)
    return markers


def apply_updated_blocks(plan: str, updated_matches: list[re.Match]) -> str:
    if not updated_matches:
        return plan
    steps = parse_plan_steps(plan)
    for match in updated_matches:
        steps[int(match.group(1))] = clean_plan_inner(match.group(2))
    return "\n".join(f"#Q_{idx}: {steps[idx]}" for idx in sorted(steps))


def apply_update_duplicate_replacements(plan: str, replacements: dict[str, str]) -> str:
    updated = plan
    for marker, replacement in replacements.items():
        replacement = (replacement or "").strip()
        if not marker.startswith("#A_") or not replacement:
            continue
        updated = updated.replace(marker, replacement)
    return updated


def replan_markers_and_target(text: str) -> tuple[list[str], str]:
    steps = parse_plan_steps(text)
    if not steps:
        return [], ""
    markers = [f"#Q_{idx}" for idx in sorted(steps)]
    target = "\n".join(f"#Q_{idx}: {steps[idx]}" for idx in sorted(steps))
    return markers, target


def parse_plan_steps(text: str) -> dict[int, str]:
    return {int(match.group(1)): match.group(2).strip() for match in Q_LINE_RE.finditer(text or "")}


def extract_think_before(text: str, *, before_pos: int) -> str:
    chosen = ""
    for match in THINK_RE.finditer(text or ""):
        if match.start() < before_pos:
            candidate = clean_text(match.group(1))
            if candidate:
                chosen = candidate
    return chosen


def update_think_anchor(content: str, marker: str, updated_matches: list[re.Match]) -> int:
    anchors = [match.start() for match in updated_matches]
    marker_match = re.search(rf"^\s*{re.escape(marker)}\s*:", content or "", re.MULTILINE)
    if marker_match:
        anchors.append(marker_match.start())
    return min(anchors) if anchors else 0


def find_latest_plan_span(text: str) -> tuple[int, int, str] | None:
    stack: list[int] = []
    latest: tuple[int, int] | None = None
    for match in PLAN_SPAN_RE.finditer(text or ""):
        if match.group(0).lower() == "<plan>":
            stack.append(match.end())
        elif stack:
            latest = (stack.pop(), match.start())
    if latest is None:
        return None
    start, end = latest
    return start, end, text[start:end]


def replace_latest_plan_inner(text: str, new_inner: str) -> str:
    span = find_latest_plan_span(text)
    if span is None:
        return text
    start, end, _ = span
    return text[:start] + "\n" + new_inner.strip() + "\n" + text[end:]


def find_replan_span_start(plan: str, marker: str) -> int:
    match = re.search(rf"^\s*{re.escape(marker)}\s*:", plan or "", re.MULTILINE)
    return match.start() if match else -1


def move_past_leading_newlines(text: str, start: int) -> int:
    start = max(0, int(start))
    while start < len(text) and text[start] in "\r\n":
        start += 1
    return start


def trim_trailing_newlines(text: str, end: int, *, min_start: int) -> int:
    end = min(len(text), max(0, int(end)))
    while end > min_start and text[end - 1] in "\r\n":
        end -= 1
    return end


def block(tag: str, content: str) -> str:
    return f"<{tag}>\n{str(content).strip()}\n</{tag}>"


def merge_metadata(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    merged.update(extra or {})
    return merged


def prefixed_counter(counter: Counter, prefix: str) -> Counter:
    return Counter({f"{prefix}{key}": value for key, value in counter.items()})


def strip_role_markers(text: str) -> str:
    return (
        str(text or "")
        .replace(HUMAN_ROLE_MARKER, "")
        .replace(ASSISTANT_ROLE_MARKER, "")
        .replace("<role>HUMAN</role>", "")
        .replace("<role>ASSISTANT</role>", "")
        .replace("<|role_end|>", "")
    )


def count_teacher_only_artifacts(text: str) -> Counter:
    counts = Counter()
    raw_subanswer_lines = len(SUBANSWER_RE.findall(text or ""))
    updated_q_blocks = len(UPDATED_RE.findall(text or ""))
    if raw_subanswer_lines:
        counts["raw_subanswer_lines"] = raw_subanswer_lines
    if updated_q_blocks:
        counts["updated_q_blocks"] = updated_q_blocks
    return counts


def strip_teacher_only_artifacts(text: str) -> str:
    text = SUBANSWER_RE.sub("", text or "")
    text = UPDATED_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_answer_direct_target(text: str) -> bool:
    text = str(text or "").strip()
    if re.fullmatch(r"<answer>\n.+?\n</answer>", text, flags=re.DOTALL | re.IGNORECASE):
        return True
    return bool(re.fullmatch(r"<think>\n.+?\n</think>\n<answer>\n.+?\n</answer>", text, flags=re.DOTALL | re.IGNORECASE))


def clean_plan_inner(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text or "").splitlines() if line.strip()).strip()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def valid_target_ids(target_ids: list[int], max_target_tokens: int) -> bool:
    return bool(target_ids) and len(target_ids) <= max(1, int(max_target_tokens))


def tokenize(tokenizer: Any, text: str) -> list[int]:
    return [int(item) for item in tokenizer.encode(text, add_special_tokens=False)]


def render_ids(tokenizer: Any, token_ids: Iterable[int]) -> str:
    mask_text = tokenizer.decode([MASK_TOKEN_ID], skip_special_tokens=False)
    text = tokenizer.decode([int(item) for item in token_ids], skip_special_tokens=False)
    return text.replace(mask_text, "<mask>").replace(EXPAND_TOKEN_TEXT, "<expand>")


def even_partition_ids(ids: list[int], parts: int) -> list[list[int]]:
    base, extra = divmod(len(ids), parts)
    out = []
    cursor = 0
    for idx in range(parts):
        size = base + (1 if idx < extra else 0)
        out.append(ids[cursor: cursor + size])
        cursor += size
    return out


def unique_ints(values: Iterable[int]) -> list[int]:
    out = []
    seen = set()
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def sample_dreamon_groups(target_ids: list[int], rng: random.Random, low: float, high: float, merge_prob: float) -> list[tuple[str, list[int]]]:
    low = max(0.0, min(1.0, low))
    high = max(0.0, min(1.0, high))
    if low > high:
        low, high = high, low
    mask_prob = rng.uniform(low, high)
    flags = [rng.random() < mask_prob for _ in target_ids]
    if not any(flags):
        flags[rng.randrange(len(flags))] = True
    groups = []
    idx = 0
    while idx < len(target_ids):
        if not flags[idx]:
            groups.append(("resolved", [target_ids[idx]]))
            idx += 1
            continue
        segment = [target_ids[idx]]
        idx += 1
        while idx < len(target_ids) and flags[idx] and rng.random() < merge_prob:
            segment.append(target_ids[idx])
            idx += 1
        groups.append(("mask", segment))
    return groups


def audit_records(action_records: list[dict[str, Any]], edit_records: list[dict[str, Any]]) -> dict[str, Any]:
    action_bad = Counter()
    for record in action_records:
        target = record.get("target_text", "")
        if re.search(r"^\s*#A_\d+\s*:", strip_update_replan_content(target), re.MULTILINE):
            action_bad["raw_answer_helper_in_action"] += 1
        if UPDATED_RE.search(target):
            action_bad["updated_q_block_in_action"] += 1
        if "<mask>" in target or "<expand>" in target or EXPAND_TOKEN_TEXT in target:
            action_bad["internal_token_in_action"] += 1
        if not has_terminal_action(target):
            action_bad["missing_terminal_action"] += 1
    edit_bad = Counter()
    for record in edit_records:
        if not any(int(item) for item in record.get("loss_mask", [])):
            edit_bad["no_active_loss"] += 1
        if len(record.get("input_token_ids", [])) != len(record.get("label_token_ids", [])):
            edit_bad["length_mismatch"] += 1
    return {"action_bad": action_bad, "edit_bad": edit_bad}


def strip_update_replan_content(text: str) -> str:
    text = re.sub(r"<update>.*?</update>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<replan>.*?</replan>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def has_terminal_action(text: str) -> bool:
    return bool(SEARCH_RE.search(text or "") or ANSWER_RE.search(text or "") or re.search(r"<update>.*?</update>", text or "", re.DOTALL | re.IGNORECASE) or REPLAN_RE.search(text or ""))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(flatten_record(record), ensure_ascii=False) + "\n")


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for key in ("context", "metadata", "segments"):
        if key in out and not isinstance(out[key], (str, int, float, bool, type(None))):
            out[key] = json.dumps(to_jsonable(out[key]), ensure_ascii=False, sort_keys=True)
    return to_jsonable(out)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
