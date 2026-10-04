"""Group-normalized answer F1 with clean, soft-invalid, and hard-invalid format gates.

Answer scoring is adapted from TraceRL/reward/qa_reward.py."""

from __future__ import annotations

import re
import statistics
import string
from collections import Counter
from typing import Any

from diffuplan_core.metrics import classify_format as _shared_classify_format


# ----------------------------------------------------------------------------- F1
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def _bool_mapping(s: str) -> str:
    if s == "True":
        return "yes"
    if s == "False":
        return "no"
    return s


def f1_score(prediction: str, ground_truth: str) -> float:
    np_ = normalize_answer(_bool_mapping(prediction))
    ng = normalize_answer(_bool_mapping(ground_truth))
    if np_ in ["yes", "no", "noanswer"] and np_ != ng:
        return 0.0
    if ng in ["yes", "no", "noanswer"] and np_ != ng:
        return 0.0
    pt = np_.split()
    gt = ng.split()
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return (2 * precision * recall) / (precision + recall)


def compute_f1(prediction: str, ground_truth: Any) -> float:
    if not prediction:
        return 0.0
    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target", ground_truth.get("answer", []))
    else:
        targets = ground_truth
    if isinstance(targets, str):
        targets = [targets]
    if not targets:
        return 0.0
    return max(f1_score(prediction, str(t)) for t in targets)


def classify_format(episode: dict[str, Any]) -> str:
    """Return 'clean' | 'soft' | 'hard' using the same classifier as eval metrics."""
    return _shared_classify_format(episode)


# ----------------------------------------------------------------- GRPO reward
def compute_group_rewards(
    episodes: list[dict[str, Any]],
    *,
    gate_clean: float = 1.0,
    gate_soft: float = 0.5,
    gate_hard: float = 0.0,
    adv_eps: float = 1e-4,
) -> dict[str, Any]:
    """Attach reward (group-normalized gated F1), loss_weight, and diagnostics in place.

    Episodes are grouped by input_index (k samples per prompt). Each episode gets:
      raw_f1, format_class, gate, gated_score, reward (group-centered/normalized), loss_weight.
    """
    gates = {"clean": gate_clean, "soft": gate_soft, "hard": gate_hard}

    # 1) per-episode gated score
    for ep in episodes:
        f1 = compute_f1(str(ep.get("prediction", "") or ""), ep.get("ground_truth", ep.get("reference", [])))
        fmt = classify_format(ep)
        ep["raw_f1"] = float(f1)
        ep["format_class"] = fmt
        ep["gate"] = float(gates.get(fmt, 0.0))
        ep["gated_score"] = float(f1) * float(gates.get(fmt, 0.0))

    # 2) group by prompt, GRPO-normalize
    groups: dict[int, list[dict[str, Any]]] = {}
    for ep in episodes:
        groups.setdefault(int(ep.get("input_index", 0)), []).append(ep)

    for _, group in groups.items():
        scores = [ep["gated_score"] for ep in group]
        if len(scores) > 1:
            mean = statistics.mean(scores)
            stdev = statistics.stdev(scores)
            for ep in group:
                ep["reward"] = (ep["gated_score"] - mean) / (stdev + adv_eps)
                # zero-variance groups carry no signal -> zero loss weight (TraceRL convention)
                ep["loss_weight"] = 0.0 if stdev < 1e-8 else 1.0
        else:
            for ep in group:
                ep["reward"] = float(ep["gated_score"])
                ep["loss_weight"] = 1.0

    return _summarize(episodes)


def _summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(episodes))

    def _avg(key):
        return float(sum(float(ep.get(key, 0.0) or 0.0) for ep in episodes) / n)

    multi_hop = [ep for ep in episodes if int(ep.get("retrieval_used", 0) or 0) >= 2]
    collapse = sum(
        1 for ep in multi_hop if (int(ep.get("num_update", 0) or 0) + int(ep.get("num_replan", 0) or 0)) == 0
    )
    return {
        "mean_f1": _avg("raw_f1"),
        "mean_gated_score": _avg("gated_score"),
        "mean_reward": _avg("reward"),
        "hard_format_frac": sum(1 for ep in episodes if ep.get("format_class") == "hard") / n,
        "soft_format_frac": sum(1 for ep in episodes if ep.get("format_class") == "soft") / n,
        "clean_format_frac": sum(1 for ep in episodes if ep.get("format_class") == "clean") / n,
        "mean_update_count": _avg("num_update"),
        "mean_replan_count": _avg("num_replan"),
        "mean_search_count": _avg("num_search"),
        "mean_retrieval_used": _avg("retrieval_used"),
        "protocol_collapse_frac": (collapse / max(1, len(multi_hop))) if multi_hop else 0.0,
        "num_episodes": len(episodes),
    }
