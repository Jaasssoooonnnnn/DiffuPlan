"""Reusable inference engine wrapper around the editable-plan rollout loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .io_utils import InputItem, make_output_record
from .rollout import RolloutConfig, run_editable_plan_rollout
from .search_client import SearchClient


@dataclass
class EnginePerf:
    batch_elapsed_s: float = 0.0
    generated_tokens: int = 0
    retrieval_used: int = 0
    rows: int = 0
    exceptions: int = 0


@dataclass
class EditablePlanInferenceEngine:
    """Small reusable engine for inference/eval/RL collection drivers."""

    model: Any
    tokenizer: Any
    search_client: Optional[SearchClient]
    config: RolloutConfig
    action_dinfer_engine: Any = None
    edit_dinfer_engine: Any = None
    last_perf: EnginePerf = field(default_factory=EnginePerf)

    def generate_one(
        self,
        *,
        item: InputItem,
        sample_index: int = 0,
        fail_on_error: bool = False,
    ) -> dict[str, Any]:
        t0 = time.time()
        try:
            result = run_editable_plan_rollout(
                model=self.model,
                tokenizer=self.tokenizer,
                question=item.question,
                search_client=self.search_client,
                config=self.config,
                action_dinfer_engine=self.action_dinfer_engine,
                edit_dinfer_engine=self.edit_dinfer_engine,
            )
            return make_output_record(
                item=item,
                sample_index=sample_index,
                result=result,
                elapsed_s=time.time() - t0,
            )
        except Exception as exc:
            if fail_on_error:
                raise
            return {
                "input_index": int(item.input_index),
                "sample_index": int(sample_index),
                "dataset_name": item.dataset_name,
                "question": item.question,
                "reference": item.reference,
                "prediction": "",
                "retrieval_used": 0,
                "stop_reason": "exception",
                "generated_tokens_model_only": 0,
                "response_length": 0,
                "elapsed_s": time.time() - t0,
                "prompt": "",
                "full_output": "",
                "latest_plan": "",
                "actions": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    def generate_rollouts(
        self,
        *,
        items: list[InputItem],
        k_sample: int = 1,
        fail_on_error: bool = False,
        progress: bool = False,
        print_limit: int = 0,
    ) -> list[dict[str, Any]]:
        t_batch = time.time()
        records: list[dict[str, Any]] = []
        total = len(items) * max(1, int(k_sample))

        for item_idx, item in enumerate(items):
            for sample_index in range(max(1, int(k_sample))):
                record = self.generate_one(
                    item=item,
                    sample_index=sample_index,
                    fail_on_error=fail_on_error,
                )
                records.append(record)
                if progress and len(records) <= max(0, int(print_limit)):
                    print(
                        f"[{len(records)}/{total}] [{record['input_index']}:{sample_index}] "
                        f"stop={record['stop_reason']} retrieval={record['retrieval_used']} "
                        f"answer={record['prediction']}",
                        flush=True,
                    )
            if progress and len(items) > 1:
                print(f"[Progress] item {item_idx + 1}/{len(items)} complete", flush=True)

        self.last_perf = EnginePerf(
            batch_elapsed_s=time.time() - t_batch,
            generated_tokens=sum(int(r.get("generated_tokens_model_only", 0) or 0) for r in records),
            retrieval_used=sum(int(r.get("retrieval_used", 0) or 0) for r in records),
            rows=len(records),
            exceptions=sum(1 for r in records if r.get("stop_reason") == "exception"),
        )
        return records

    def get_last_perf(self) -> dict[str, Any]:
        return {
            "batch_elapsed_s": float(self.last_perf.batch_elapsed_s),
            "generated_tokens": int(self.last_perf.generated_tokens),
            "retrieval_used": int(self.last_perf.retrieval_used),
            "rows": int(self.last_perf.rows),
            "exceptions": int(self.last_perf.exceptions),
        }
