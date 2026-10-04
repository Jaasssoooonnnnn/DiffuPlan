"""Lightweight QA metrics and control-tag format checks."""

from __future__ import annotations

import collections
import re
import string
from typing import Any


CONTROL_TAGS = ("think", "plan", "search", "update", "replan", "information", "answer")
OPENING_CONTROL_TAG_RE = re.compile(r"<(think|plan|search|update|replan|information|answer)>")
BROKEN_CLOSE_FRAGMENT_RE = re.compile(
    r"</</|</(?:think|plan|search|update|replan|answer|information)\.|"
    r"</(?:think|plan|search|update|replan|answer|information)\b(?!>)"
)
EMPTY_SEARCH_RE = re.compile(r"<search>\s*</search>")
ANSWER_SPAN_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FIRST_TAG_RE = re.compile(r"\s*<([A-Za-z_][A-Za-z0-9_]*)>")


def extract_targets(reference: Any) -> list[str]:
    if isinstance(reference, dict):
        reference = reference.get("target", reference.get("answer", []))
    if isinstance(reference, str):
        return [reference]
    if isinstance(reference, (list, tuple)):
        return [str(item) for item in reference if str(item)]
    return []


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in value)

    value = str(text or "").replace("_", " ").lower()
    return " ".join(remove_articles(remove_punc(value)).split())


def exact_match(prediction: str, target: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(target)


def sub_exact_match(prediction: str, target: str) -> bool:
    """True if the normalized reference appears in the normalized prediction."""
    pred = normalize_answer(prediction)
    gold = normalize_answer(target)
    return bool(gold) and gold in pred


def f1_score(prediction: str, target: str) -> float:
    pred = normalize_answer(_bool_mapping(prediction))
    gold = normalize_answer(_bool_mapping(target))
    if pred in {"yes", "no", "noanswer"} and pred != gold:
        return 0.0
    if gold in {"yes", "no", "noanswer"} and pred != gold:
        return 0.0
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(1, len(pred_tokens))
    recall = num_same / max(1, len(gold_tokens))
    return float((2.0 * precision * recall) / (precision + recall))


def _bool_mapping(text: str) -> str:
    if text == "True":
        return "yes"
    if text == "False":
        return "no"
    return text


def attach_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        targets = extract_targets(record.get("reference", []))
        prediction = str(record.get("prediction", "") or "")
        record["em"] = float(any(exact_match(prediction, target) for target in targets)) if targets else 0.0
        record["subem"] = float(any(sub_exact_match(prediction, target) for target in targets)) if targets else 0.0
        record["f1"] = max((f1_score(prediction, target) for target in targets), default=0.0)
        record["format_class"] = classify_format(record)
        record["format_error"] = bool(record["format_class"] != "clean")
    return summarize_records(records)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {"overall": records}
    for record in records:
        dataset_name = str(record.get("dataset_name", "") or "")
        if dataset_name:
            buckets.setdefault(dataset_name, []).append(record)

    summary: dict[str, Any] = {}
    for name, rows in buckets.items():
        hard_count = sum(1 for row in rows if row.get("format_class") == "hard")
        soft_count = sum(1 for row in rows if row.get("format_class") == "soft")
        summary[name] = {
            "count": float(len(rows)),
            "mean_em": _avg([float(row.get("em", 0.0) or 0.0) for row in rows]),
            "mean_subem": _avg([float(row.get("subem", 0.0) or 0.0) for row in rows]),
            "mean_f1": _avg([float(row.get("f1", 0.0) or 0.0) for row in rows]),
            "avg_retrieval_used": _avg([float(row.get("retrieval_used", 0.0) or 0.0) for row in rows]),
            "avg_response_length": _avg([float(row.get("response_length", 0.0) or 0.0) for row in rows]),
            "avg_generated_tokens_model_only": _avg(
                [float(row.get("generated_tokens_model_only", 0.0) or 0.0) for row in rows]
            ),
            "format_error_frac": ((hard_count + soft_count) / len(rows)) if rows else 0.0,
            "hard_format_count": float(hard_count),
            "soft_format_count": float(soft_count),
        }
    return summary


def classify_format(record: dict[str, Any]) -> str:
    format_text = _format_check_text(record)
    prediction = str(record.get("prediction", "") or "")
    stop_reason = str(record.get("stop_reason", "") or "")
    actions = record.get("actions", []) or []
    counts = count_control_tags(format_text)

    hard = (
        stop_reason in {"no_complete_action", "unknown_action"}
        or not has_balanced_control_tags(counts)
        or bool(BROKEN_CLOSE_FRAGMENT_RE.search(format_text))
        or (_first_tag(format_text) != "plan")
        or _has_generated_information(actions)
        or _has_bad_marker_action(actions)
        or not prediction
        or "<" in prediction
        or ">" in prediction
        or counts.get("answer_open", 0) > 1
        or counts.get("answer_close", 0) > 1
        or any("<search>" in span or "</search>" in span for span in ANSWER_SPAN_RE.findall(format_text))
    )
    if hard:
        return "hard"

    opening_tags = OPENING_CONTROL_TAG_RE.findall(format_text)
    if opening_tags and opening_tags[0] not in {"think", "plan"}:
        return "soft"
    if bool(EMPTY_SEARCH_RE.search(format_text)):
        return "soft"
    if has_repeated_search_without_rethink(opening_tags):
        return "soft"
    return "clean"


def _format_check_text(record: dict[str, Any]) -> str:
    actions = record.get("actions")
    if isinstance(actions, list):
        raw_actions = [
            str(action.get("raw", ""))
            for action in actions
            if isinstance(action, dict) and action.get("raw") is not None
        ]
        if raw_actions:
            return "\n".join(raw_actions)
    return str(record.get("full_output", "") or "")


def _first_tag(text: str) -> str:
    match = FIRST_TAG_RE.match(text or "")
    return match.group(1) if match else ""


def _has_generated_information(actions: Any) -> bool:
    if not isinstance(actions, list):
        return False
    for action in actions:
        if not isinstance(action, dict):
            continue
        raw = str(action.get("raw", "") or "")
        if "<information>" in raw or "</information>" in raw:
            return True
    return False


def _marker_list_ok(content: str, prefix: str) -> bool:
    items = [item for item in re.split(r"[,\s]+", (content or "").strip()) if item]
    if not items:
        return False
    return all(re.fullmatch(rf"#{prefix}_\d+", item) for item in items)


def _has_bad_marker_action(actions: Any) -> bool:
    if not isinstance(actions, list):
        return False
    for action in actions:
        if not isinstance(action, dict):
            continue
        kind = action.get("kind")
        content = str(action.get("content", "") or "")
        if kind == "update" and not _marker_list_ok(content, "A"):
            return True
        if kind == "replan" and not _marker_list_ok(content, "Q"):
            return True
    return False


def count_control_tags(text: str) -> dict[str, int]:
    return {
        f"{tag}_open": text.count(f"<{tag}>")
        for tag in CONTROL_TAGS
    } | {
        f"{tag}_close": text.count(f"</{tag}>")
        for tag in CONTROL_TAGS
    }


def has_balanced_control_tags(counts: dict[str, int]) -> bool:
    return all(counts.get(f"{tag}_open", 0) == counts.get(f"{tag}_close", 0) for tag in CONTROL_TAGS)


def has_repeated_search_without_rethink(opening_tags: list[str]) -> bool:
    prev = None
    for tag in opening_tags:
        if tag == "information":
            continue
        if prev == "search" and tag == "search":
            return True
        prev = tag
    return False


def _avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
