#!/usr/bin/env python3
"""Collect teacher trajectories for DiffuPlan SFT.

A teacher model served behind an OpenAI-compatible endpoint writes an indexed
plan, searches the retriever, records subanswers (#A_i), rewrites dependent
steps (<updated_#Q_j>), optionally replans the unresolved suffix, and answers.
Trajectories that pass the acceptance checks are written as teacher_trace_text
records, the input format of build_sft_from_canonical_traces.py:

  traces.jsonl         complete trajectories        (--full-trace-input)
  answer_direct.jsonl  trajectories that reached the search budget and were
                       asked to answer directly      (--answer-direct-trace-input)

Questions rejected in one pass are resampled in the next pass.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
import yaml


SCHEMA_VERSION = "teacher_trace_text"
HUMAN_ROLE_MARKER = "<|role_end|><role>HUMAN</role>"
ASSISTANT_ROLE_MARKER = "<|role_end|><role>ASSISTANT</role>"
BUDGET_MESSAGE = "The action budget is exhausted. I should now give the final answer between <answer> and </answer>."

PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
REPLAN_RE = re.compile(r"<replan>(.*?)</replan>", re.DOTALL | re.IGNORECASE)
UPDATED_RE = re.compile(r"<updated_#Q_(\d+)>(.*?)</updated_#Q_\1>", re.DOTALL | re.IGNORECASE)
SUBANSWER_RE = re.compile(r"^\s*#A_(\d+)\s*:\s*(.+?)\s*$", re.MULTILINE)
STEP_RE = re.compile(r"^\s*#Q_(\d+)\s*:\s*(.+?)\s*$", re.MULTILINE)
ANSWER_REF_RE = re.compile(r"#A_(\d+)")
ENV_TAG_RE = re.compile(r"</?(?:information|observation|tool_call|tool_response)\b", re.IGNORECASE)
BALANCED_TAGS = ("plan", "think", "search", "replan", "answer")


TRACE_PROMPT = """Answer the question by planning, searching, and keeping the plan up to date. Use only the tags described below; do not output JSON or markdown.

Question: {question}

1. Plan. Start with an ordered plan of atomic, answerable sub-questions. A later step may refer to the answer of an earlier step as #A_i; never refer to a later step.
<plan>
#Q_1: first sub-question
#Q_2: next sub-question, which may use #A_1
</plan>
For a comparison, find each compared value in its own step and end with a step that compares the earlier answers. Keep the answer type the question asks for, such as nationality rather than country or birth date rather than death date.

2. Search. Before every search, write one short, non-empty sentence explaining it, then the query:
<think>
I need to identify the first entity.
</think>
<search>
query
</search>
After </search>, stop. The environment returns the retrieved passages between <information> and </information>. Never write <information> yourself.

3. Record answers. When the retrieved evidence answers the current step, write
<think>
The evidence states the answer to the current step.
</think>
#A_i: concise answer to #Q_i
Use only facts supported by the retrieved passages. If the evidence is insufficient, search again with a better query.

4. Update. After an #A_i line, rewrite each unresolved later step that depends on it, replacing the reference with the concrete answer:
<updated_#Q_j>
rewritten #Q_j
</updated_#Q_j>
Never modify a solved step.

5. Replan. Only when the current route has failed, because different queries keep returning no useful evidence, the evidence contradicts the plan, or the remaining steps are wrong, replace the unresolved steps from the current step to the end of the plan while keeping solved steps:
<replan>
#Q_i: replacement step
#Q_k: replacement final step
</replan>
Then continue with <think> and <search>. Do not replan only to reword a step or to fill in an answer you already found.

6. Answer. When every step has an #A_i line, finish with only the short answer span:
<answer>
short answer
</answer>

You may use at most {max_searches} searches and {max_replans} replan. Every reply must end with exactly one <search> block or one <answer> block. Your first reply must contain the <plan>, a <think>, and a <search>."""

FOLLOWUP_PROMPT = """Current plan:
{plan_state}

Searches used: {searches}/{max_searches}. Continue from the latest information. Do not write a new <plan>. Begin with a non-empty <think>. If the evidence answers the current step, record #A_i and update dependent steps; otherwise search with a better query, or replan if the route has failed."""

INVALID_PROMPT = """Your previous reply did not follow the protocol ({reason}). Reply again with valid tags: a non-empty <think> immediately followed by <search>, or the missing #A_i lines followed by <answer>. Do not repeat the plan."""

INCOMPLETE_PROMPT = """These planned steps have no #A_i line yet: {missing}. Record their answers from the retrieved evidence, or search for them, before giving <answer>."""


# --------------------------------------------------------------------------- clients
class TeacherClient:
    """OpenAI-compatible chat client that round-robins across server URLs."""

    def __init__(self, cfg: dict[str, Any]):
        self.model = cfg["model"]
        self.urls = [url.rstrip("/") for url in cfg["base_urls"]]
        self.api_key = cfg.get("api_key") or "EMPTY"
        self.body = {
            "temperature": cfg.get("temperature", 0.5),
            "top_p": cfg.get("top_p", 0.8),
            "max_tokens": cfg.get("max_tokens", 1024),
            **(cfg.get("extra_body") or {}),
        }
        self.timeout = cfg.get("timeout", 120)
        self.retries = cfg.get("max_retries", 3)
        self._cycle = itertools.cycle(self.urls)
        self._lock = threading.Lock()

    def complete(self, messages: list[dict[str, str]]) -> str:
        with self._lock:
            url = next(self._cycle)
        payload = {"model": self.model, "messages": messages, **self.body}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.retries):
            try:
                response = requests.post(f"{url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"] or ""
            except Exception:
                if attempt + 1 == self.retries:
                    raise
                time.sleep(2.0 * (attempt + 1))
        return ""


class Retriever:
    """Client for Diffuplan/retrieval_server.py; returns passages as Doc i(Title: ...) text."""

    def __init__(self, cfg: dict[str, Any]):
        self.url = cfg["url"]
        self.topk = cfg.get("topk", 3)
        self.max_words = cfg.get("max_obs_length", 500)
        self.timeout = cfg.get("timeout", 60)

    def search(self, query: str) -> str:
        payload = {"queries": [query], "topk": self.topk, "return_scores": True}
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        docs = (response.json().get("result") or [[]])[0]
        rows = []
        for idx, item in enumerate(docs, start=1):
            contents = (item.get("document", item) or {}).get("contents", "")
            title, _, body = contents.partition("\n")
            rows.append(f"Doc {idx}(Title: {title.strip().strip(chr(34))}) {body.strip()}".strip())
        text = "\n".join(rows)
        words = text.split()
        return " ".join(words[: self.max_words]) if len(words) > self.max_words else text


# --------------------------------------------------------------------------- metrics
def normalize_answer(text: str) -> str:
    text = "".join(ch for ch in str(text or "").lower() if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = sum((collections.Counter(pred) & collections.Counter(ref)).values())
    if not pred or not ref or common == 0:
        return float(pred == ref)
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def qa_scores(prediction: str, golds: list[str]) -> dict[str, float]:
    golds = golds or [""]
    return {
        "em": max(float(normalize_answer(prediction) == normalize_answer(g)) for g in golds),
        "subem": max(float(bool(normalize_answer(g)) and normalize_answer(g) in normalize_answer(prediction)) for g in golds),
        "f1": max(token_f1(prediction, g) for g in golds),
    }


# --------------------------------------------------------------------------- plan state
@dataclass
class Plan:
    steps: dict[int, str] = field(default_factory=dict)
    answers: dict[int, str] = field(default_factory=dict)
    replans: int = 0

    def unresolved(self) -> list[int]:
        return [sid for sid in sorted(self.steps) if sid not in self.answers]

    def render(self) -> str:
        lines = []
        for sid in sorted(self.steps):
            lines.append(f"#Q_{sid}: {self.steps[sid]}")
            if sid in self.answers:
                lines.append(f"#A_{sid}: {self.answers[sid]}")
        return "\n".join(lines)


def parse_steps(block: str) -> dict[int, str]:
    return {int(m.group(1)): m.group(2).strip() for m in STEP_RE.finditer(block or "")}


def steps_problem(steps: dict[int, str], first_id: int = 1) -> Optional[str]:
    ids = sorted(steps)
    if not ids or ids != list(range(first_id, first_id + len(ids))):
        return "non_contiguous_steps"
    for sid, question in steps.items():
        if any(int(ref) >= sid for ref in ANSWER_REF_RE.findall(question)):
            return "forward_answer_reference"
    return None


def protocol_problem(chunk: str, *, first_turn: bool) -> Optional[str]:
    for tag in BALANCED_TAGS:
        if len(re.findall(rf"<{tag}>", chunk, re.I)) != len(re.findall(rf"</{tag}>", chunk, re.I)):
            return f"unbalanced_{tag}"
    if ENV_TAG_RE.search(chunk):
        return "generated_environment_tag"
    if first_turn != bool(PLAN_RE.search(chunk)):
        return "plan_expected_in_first_reply_only"
    search = SEARCH_RE.search(chunk)
    if first_turn and not search:
        return "first_reply_without_search"
    if search:
        thinks = [m for m in THINK_RE.finditer(chunk) if m.end() <= search.start()]
        if not thinks or not thinks[-1].group(1).strip():
            return "missing_think_before_search"
        if not search.group(1).strip():
            return "empty_search"
        return None
    if ANSWER_RE.search(chunk) and ANSWER_RE.search(chunk).group(1).strip():
        return None
    return "no_terminal_action"


def first_terminal(text: str) -> str:
    ends = [m.end() for m in (SEARCH_RE.search(text or ""), ANSWER_RE.search(text or "")) if m]
    return (text or "")[: min(ends)].strip() if ends else (text or "").strip()


def apply_edits(plan: Plan, chunk: str, max_replans: int) -> Optional[str]:
    """Apply #A_i, <updated_#Q_j>, and <replan> in reply order; return a violation or None."""
    ops = [(m.start(), "answer", m) for m in SUBANSWER_RE.finditer(chunk)]
    ops += [(m.start(), "update", m) for m in UPDATED_RE.finditer(chunk)]
    ops += [(m.start(), "replan", m) for m in REPLAN_RE.finditer(chunk)]
    for _, kind, match in sorted(ops, key=lambda op: op[0]):
        sid = int(match.group(1)) if kind != "replan" else None
        if kind == "answer":
            if sid not in plan.steps:
                return f"answer_for_unknown_step:{sid}"
            plan.answers[sid] = match.group(2).strip()
        elif kind == "update":
            text = match.group(2).strip()
            if sid not in plan.steps or sid in plan.answers or not text:
                return f"invalid_update:{sid}"
            plan.steps[sid] = text
        else:
            plan.replans += 1
            replacement = parse_steps(match.group(1))
            unresolved = plan.unresolved()
            if plan.replans > max_replans or not replacement or not unresolved:
                return "invalid_replan"
            start = min(replacement)
            if start not in unresolved or steps_problem(replacement, first_id=start):
                return "invalid_replan_suffix"
            plan.steps = {sid: q for sid, q in plan.steps.items() if sid < start or sid in plan.answers}
            plan.steps.update(replacement)
    return None


# --------------------------------------------------------------------------- sampling
class TraceSampler:
    def __init__(self, teacher: TeacherClient, retriever: Retriever, cfg: dict[str, Any]):
        self.teacher = teacher
        self.retriever = retriever
        self.max_searches = cfg.get("max_searches", 6)
        self.max_replans = cfg.get("max_replans", 1)
        self.max_repairs = cfg.get("format_repair_retries", 2)
        self.max_turns = cfg.get("max_turns", 24)

    def _reply(self, messages: list[dict[str, str]], *, first_turn: bool) -> tuple[str, Optional[str]]:
        chunk = first_terminal(self.teacher.complete(messages))
        problem = protocol_problem(chunk, first_turn=first_turn)
        for _ in range(self.max_repairs):
            if problem is None:
                break
            retry = messages + [{"role": "assistant", "content": chunk},
                                {"role": "user", "content": INVALID_PROMPT.format(reason=problem)}]
            chunk = first_terminal(self.teacher.complete(retry))
            problem = protocol_problem(chunk, first_turn=first_turn)
        return chunk, problem

    def sample(self, question: str) -> dict[str, Any]:
        prompt = TRACE_PROMPT.format(question=question, max_searches=self.max_searches, max_replans=self.max_replans)
        messages = [{"role": "user", "content": prompt}]
        pieces: list[str] = []
        plan = Plan()
        queries: set[str] = set()
        searches, budget_used, prediction = 0, False, ""

        def reject(reason: str) -> dict[str, Any]:
            return {"accepted": False, "reason": reason, "prediction": prediction, "retrieval_calls": searches}

        for turn in range(self.max_turns):
            chunk, problem = self._reply(messages, first_turn=turn == 0)
            if problem:
                return reject(f"format:{problem}")
            if turn == 0:
                plan.steps = parse_steps(PLAN_RE.search(chunk).group(1))
                if steps_problem(plan.steps):
                    return reject("invalid_plan")
            violation = apply_edits(plan, chunk, self.max_replans)
            if violation:
                return reject(violation)

            answer = ANSWER_RE.search(chunk)
            if answer:
                missing = plan.unresolved()
                if missing and not budget_used:
                    kept = ANSWER_RE.sub("", chunk).strip()
                    if kept:
                        pieces.append(kept)
                    messages += [{"role": "assistant", "content": chunk},
                                 {"role": "user", "content": INCOMPLETE_PROMPT.format(missing=", ".join(f"#Q_{s}" for s in missing))}]
                    continue
                prediction = answer.group(1).strip()
                pieces.append(chunk)
                return {"accepted": True, "prediction": prediction, "retrieval_calls": searches,
                        "answer_direct_used": budget_used, "complete_plan": not plan.unresolved(),
                        "trace_text": "\n".join(pieces).strip()}

            if budget_used:
                return reject("search_after_budget")
            query = SEARCH_RE.search(chunk).group(1).strip()
            key = " ".join(query.lower().split())
            if key in queries:
                return reject("repeated_query")
            queries.add(key)
            pieces.append(chunk)
            messages.append({"role": "assistant", "content": chunk})
            if searches >= self.max_searches:
                budget_used = True
                pieces.append(f"{HUMAN_ROLE_MARKER}\n{BUDGET_MESSAGE}\n{ASSISTANT_ROLE_MARKER}")
                messages.append({"role": "user", "content": BUDGET_MESSAGE})
                continue
            observation = self.retriever.search(query)
            searches += 1
            info = f"<information>\n{observation}\n</information>"
            pieces.append(info)
            followup = FOLLOWUP_PROMPT.format(plan_state=plan.render(), searches=searches, max_searches=self.max_searches)
            messages.append({"role": "user", "content": f"{info}\n\n{followup}"})
        return reject("max_turns")


def accept(result: dict[str, Any], golds: list[str], cfg: dict[str, Any]) -> tuple[bool, str]:
    if not result["accepted"]:
        return False, result["reason"]
    score = qa_scores(result["prediction"], golds)[cfg.get("metric", "f1")]
    if score <= cfg.get("threshold", 0.65):
        return False, "answer_below_threshold"
    if cfg.get("require_retrieval", True) and result["retrieval_calls"] == 0:
        return False, "no_retrieval"
    if not result["answer_direct_used"] and not result["complete_plan"]:
        return False, "unfinished_plan"
    return True, ""


# --------------------------------------------------------------------------- driver
def load_questions(path: Path, data_source: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
    records = []
    for idx, row in enumerate(rows):
        golds = next((row[k] for k in ("gold_answers", "golden_answers", "answer", "answers") if row.get(k)), None)
        if not row.get("question") or golds is None:
            continue
        rid = str(row.get("id", row.get("_id", idx)))
        records.append({"id": f"{data_source}:{rid}", "question": str(row["question"]),
                        "gold_answers": [str(g) for g in (golds if isinstance(golds, list) else [golds])]})
    return records


def read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["source_trace_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="JSON or JSONL questions with question and answer fields.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-source", default="train")
    parser.add_argument("--passes", type=int, default=2, help="Each later pass resamples questions rejected by the previous one.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sampler = TraceSampler(TeacherClient(cfg["teacher"]), Retriever(cfg["retriever"]), cfg.get("sampling", {}))
    acceptance = cfg.get("acceptance", {})
    workers = args.max_workers or cfg.get("sampling", {}).get("max_workers", 8)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = {"full": out / "traces.jsonl", "direct": out / "answer_direct.jsonl", "rejected": out / "rejected.jsonl"}
    done = read_ids(paths["full"]) | read_ids(paths["direct"])
    pending = [r for r in load_questions(args.input, args.data_source)[: args.limit] if r["id"] not in done]
    summary = collections.Counter(resumed=len(done))
    lock = threading.Lock()

    def run(record: dict[str, Any], pass_idx: int) -> tuple[dict[str, Any], bool, str, dict[str, Any]]:
        try:
            result = sampler.sample(record["question"])
        except Exception as exc:
            result = {"accepted": False, "reason": f"error:{type(exc).__name__}", "prediction": "", "retrieval_calls": 0}
        ok, reason = accept(result, record["gold_answers"], acceptance)
        return record, ok, reason, {**result, "pass": pass_idx}

    for pass_idx in range(1, args.passes + 1):
        if not pending:
            break
        rejected = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run, record, pass_idx) for record in pending]
            for future in as_completed(futures):
                record, ok, reason, result = future.result()
                with lock:
                    if ok:
                        kind = "direct" if result["answer_direct_used"] else "full"
                        row = {
                            "schema_version": SCHEMA_VERSION,
                            "source_trace_id": record["id"],
                            "data_source": args.data_source,
                            "question": record["question"],
                            "gold_answers": record["gold_answers"],
                            "prediction": result["prediction"],
                            "scores": qa_scores(result["prediction"], record["gold_answers"]),
                            "accepted": True,
                            "retrieval_calls": result["retrieval_calls"],
                            "answer_direct_used": result["answer_direct_used"],
                            "pass": pass_idx,
                            "trace_text": result["trace_text"],
                        }
                        with paths[kind].open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        summary[f"accepted_{kind}"] += 1
                    else:
                        rejected.append(record)
                        summary[f"pass{pass_idx}_rejected:{reason.split(':')[0]}"] += 1
                        if pass_idx == args.passes:
                            with paths["rejected"].open("a", encoding="utf-8") as f:
                                f.write(json.dumps({"source_trace_id": record["id"], "reason": reason,
                                                    "prediction": result.get("prediction", "")}, ensure_ascii=False) + "\n")
        print(f"pass {pass_idx}: attempted={len(pending)} rejected={len(rejected)}", flush=True)
        pending = rejected

    (out / "summary.json").write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
