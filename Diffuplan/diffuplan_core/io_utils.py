"""Input/output helpers shared by inference and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class InputItem:
    question: str
    reference: Any = field(default_factory=list)
    input_index: int = 0
    dataset_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def load_input_payload(
    *,
    question: Optional[list[str]] = None,
    input_file: Optional[str] = None,
    dataset: Optional[list[str]] = None,
    data_dir: Optional[str | Path] = None,
    max_samples: Optional[int] = None,
) -> list[Any]:
    """Load raw rows from CLI-style question/input-file/dataset sources."""

    sources = sum(bool(x) for x in (question, input_file, dataset))
    if sources != 1:
        raise ValueError("Exactly one of question, input_file, or dataset must be provided.")

    rows: list[Any] = []
    if question:
        rows = list(question)
    elif dataset:
        if data_dir is None:
            raise ValueError("data_dir is required when loading named datasets.")
        root = Path(data_dir).expanduser().resolve()
        for dataset_name in dataset:
            path = root / f"{dataset_name}.json"
            with path.open("r") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                raise TypeError(f"Dataset payload must be a list: {path}")
            if max_samples is not None:
                payload = payload[: max(0, int(max_samples))]
            for row in payload:
                if isinstance(row, dict):
                    row = dict(row)
                    row.setdefault("_dataset_name", dataset_name)
                rows.append(row)
    else:
        path = Path(str(input_file)).expanduser().resolve()
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        elif suffix == ".txt":
            with path.open("r") as handle:
                rows = [line.strip() for line in handle if line.strip()]
        else:
            with path.open("r") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                raise TypeError(f"JSON input must be a list: {path}")
            rows = list(payload)

    if max_samples is not None and not dataset:
        rows = rows[: max(0, int(max_samples))]
    return rows


def normalize_items(rows: Iterable[Any]) -> list[InputItem]:
    items: list[InputItem] = []
    for idx, row in enumerate(rows):
        if isinstance(row, str):
            items.append(InputItem(question=row, input_index=idx))
            continue
        if not isinstance(row, dict):
            raise TypeError(f"Unsupported input row {idx}: {type(row).__name__}")
        question_text = row.get("question", row.get("prompt", ""))
        if not question_text:
            raise ValueError(f"Input row {idx} is missing question/prompt.")
        reference = row.get("answer", row.get("target", row.get("ground_truth_answer", [])))
        dataset_name = str(row.get("_dataset_name", row.get("dataset_name", "")) or "")
        items.append(
            InputItem(
                question=str(question_text),
                reference=reference,
                input_index=idx,
                dataset_name=dataset_name,
                raw=dict(row),
            )
        )
    if not items:
        raise ValueError("No input questions to run.")
    return items


def make_output_record(
    *,
    item: InputItem,
    sample_index: int,
    result: Any,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "input_index": int(item.input_index),
        "sample_index": int(sample_index),
        "dataset_name": item.dataset_name,
        "question": item.question,
        "reference": item.reference,
        "prediction": result.final_answer,
        "retrieval_used": int(result.retrieval_used),
        "stop_reason": result.stop_reason,
        "generated_tokens_model_only": int(result.generated_tokens_model_only),
        "response_length": len(result.full_output),
        "elapsed_s": float(elapsed_s),
        "prompt": result.prompt,
        "full_output": result.full_output,
        "latest_plan": result.latest_plan,
        "actions": result.actions,
    }
