#!/usr/bin/env python3
"""Build an action-upweighted DiffuPlan SFT dataset.

Each record is replicated according to its multiplicity. A fractional multiplicity
is realized deterministically: extra copies are spread evenly across the records
of that type, so a type with n records and multiplicity m yields floor(m * n) rows."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, TextIO


ACTION_FILE = "action_sft.jsonl"
EDIT_FILE = "edit_sft.jsonl"
MIXED_FILE = "mixed_sft.jsonl"
SUMMARY_FILE = "summary.json"
MIXED_FIELD_DEFAULTS = {
    "schema_version": "",
    "record_type": "",
    "source_trace_id": "",
    "source_line": 0,
    "action_type": "",
    "question": "",
    "prompt_text": "",
    "target_text": "",
    "context": "",
    "metadata": "",
    "edit_type": "",
    "state_kind": "",
    "action_text": "",
    "target_token_ids": [],
    "mask_token_id": 0,
    "expand_token_id": 0,
    "input_token_ids": [],
    "label_token_ids": [],
    "loss_mask": [],
    "plan_input_rendered": "",
    "active_mask_count": 0,
    "expand_label_count": 0,
    "segments": "",
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("multiplier must be >= 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("multiplicity must be > 0")
    return parsed


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def load_json_field(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def is_search_limit_answer_direct(rec: dict) -> bool:
    metadata = load_json_field(rec.get("metadata"))
    trace_path = str(metadata.get("trace_input_path") or "")
    return (
        metadata.get("answer_direct_reason") == "search_limit"
        or (
            metadata.get("answer_direct_used") is True
            and metadata.get("target_scope") == "answer_direct_final_segment"
        )
        or trace_path.endswith("search_limit.jsonl")
    )


def normalize_mixed_record(rec: dict) -> dict:
    normalized = {}
    for key, default in MIXED_FIELD_DEFAULTS.items():
        normalized[key] = rec.get(key, copy.deepcopy(default))
    return normalized


def write_mixed_record(out_f: TextIO, rec: dict) -> None:
    out_f.write(json.dumps(normalize_mixed_record(rec), ensure_ascii=False))
    out_f.write("\n")


def first_record(path: Path) -> dict | None:
    for rec in iter_jsonl(path):
        return rec
    return None


def count_records(path: Path, key: str | None = None) -> Counter:
    counts: Counter = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            counts["total"] += 1
            if key is not None:
                value = rec.get(key)
                if value is None:
                    raise ValueError(f"{path}:{line_no}: missing {key!r}")
                counts[str(value)] += 1
    return counts


def write_repeated_action_rows(
    *,
    action_path: Path,
    out_f: TextIO,
    multipliers: Dict[str, float],
    search_limit_multiplier: float,
) -> Counter:
    written: Counter = Counter()
    target: Dict[str, float] = {}
    emitted: Counter = Counter()
    with action_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            action_type = str(rec.get("action_type") or "")
            if not action_type:
                raise ValueError(f"{action_path}:{line_no}: missing action_type")
            multiplicity = float(multipliers.get(action_type, 1.0))
            key = action_type
            search_limit = is_search_limit_answer_direct(rec)
            if search_limit:
                multiplicity *= search_limit_multiplier
                key = f"{action_type}:search_limit"
            target[key] = target.get(key, 0.0) + multiplicity
            times = int(target[key] + 1e-9) - emitted[key]
            emitted[key] += times
            if search_limit:
                written["search_limit_answer_direct"] += times
            for _ in range(times):
                write_mixed_record(out_f, rec)
                written["total"] += 1
                written[action_type] += 1
    return written


def write_repeated_file(
    src: Path,
    out_f: TextIO,
    multiplier: int,
    *,
    skip_first_once: bool = False,
) -> int:
    total = 0
    for repeat_idx in range(multiplier):
        skipped_first = False
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if skip_first_once and repeat_idx == 0 and not skipped_first:
                    skipped_first = True
                    continue
                rec = json.loads(line)
                write_mixed_record(out_f, rec)
                total += 1
    return total


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replan-multiplier", type=positive_float, default=3.0)
    parser.add_argument("--update-multiplier", type=positive_float, default=1.5)
    parser.add_argument("--search-multiplier", type=positive_float, default=1.0)
    parser.add_argument("--answer-multiplier", type=positive_float, default=1.0)
    parser.add_argument(
        "--search-limit-multiplier",
        type=positive_float,
        default=3.0,
        help="Multiplier applied to answer-direct records built from search_limit.jsonl.",
    )
    parser.add_argument("--edit-multiplier", type=positive_int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    action_path = input_dir / ACTION_FILE
    edit_path = input_dir / EDIT_FILE
    summary_path = input_dir / SUMMARY_FILE
    for path in [action_path, edit_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    shutil.copy2(action_path, output_dir / ACTION_FILE)
    shutil.copy2(edit_path, output_dir / EDIT_FILE)

    multipliers = {
        "search": args.search_multiplier,
        "update": args.update_multiplier,
        "answer": args.answer_multiplier,
        "replan": args.replan_multiplier,
    }
    base_action_counts = count_records(action_path, "action_type")
    base_edit_counts = count_records(edit_path, "edit_type")

    mixed_path = output_dir / MIXED_FILE
    with mixed_path.open("w", encoding="utf-8") as out_f:
        first_edit_record = first_record(edit_path)
        weighted_edit_records = 0
        if first_edit_record is not None:
            write_mixed_record(out_f, first_edit_record)
            weighted_edit_records = 1
        weighted_action_counts = write_repeated_action_rows(
            action_path=action_path,
            out_f=out_f,
            multipliers=multipliers,
            search_limit_multiplier=args.search_limit_multiplier,
        )
        weighted_edit_records += write_repeated_file(
            edit_path,
            out_f,
            args.edit_multiplier,
            skip_first_once=first_edit_record is not None,
        )

    weighted_total = weighted_action_counts["total"] + weighted_edit_records
    summary = {
        "schema_version": "diffuplan_sft_action_upweighted",
        "source_dir": str(input_dir),
        "source_summary": load_summary(summary_path),
        "multipliers": {
            "action": multipliers,
            "search_limit_answer_direct": args.search_limit_multiplier,
            "edit": args.edit_multiplier,
        },
        "base_counts": {
            "action_records": base_action_counts["total"],
            "search_limit_answer_direct_records": sum(
                1 for rec in iter_jsonl(action_path) if is_search_limit_answer_direct(rec)
            ),
            "edit_records": base_edit_counts["total"],
            "action_type_counts": {
                key: base_action_counts[key]
                for key in sorted(base_action_counts)
                if key != "total"
            },
            "edit_type_counts": {
                key: base_edit_counts[key]
                for key in sorted(base_edit_counts)
                if key != "total"
            },
        },
        "weighted_counts": {
            "action_records": weighted_action_counts["total"],
            "search_limit_answer_direct_records": weighted_action_counts[
                "search_limit_answer_direct"
            ],
            "edit_records": weighted_edit_records,
            "mixed_records": weighted_total,
            "action_type_counts": {
                key: weighted_action_counts[key]
                for key in sorted(weighted_action_counts)
                if key not in {"total", "search_limit_answer_direct"}
            },
        },
        "weighted_fractions": {
            "action": weighted_action_counts["total"] / weighted_total,
            "edit": weighted_edit_records / weighted_total,
            "replan": weighted_action_counts["replan"] / weighted_total,
            "update": weighted_action_counts["update"] / weighted_total,
            "search_limit_answer_direct": (
                weighted_action_counts["search_limit_answer_direct"] / weighted_total
            ),
        },
        "outputs": {
            "action_sft": str(output_dir / ACTION_FILE),
            "edit_sft": str(output_dir / EDIT_FILE),
            "mixed_sft": str(mixed_path),
        },
    }
    with (output_dir / SUMMARY_FILE).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(summary["weighted_counts"], ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(summary["weighted_fractions"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
