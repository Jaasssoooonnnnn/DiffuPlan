#!/usr/bin/env python
"""Standalone LLaDA2-mini editable-plan inference runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DIFFUPLAN_ROOT = SCRIPT_DIR.parent
if str(DIFFUPLAN_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUPLAN_ROOT))

from diffuplan_core.model_utils import initialize_expand_token_mean, resolve_expand_token
from diffuplan_core.engine import EditablePlanInferenceEngine
from diffuplan_core.io_utils import (
    InputItem,
    load_input_payload,
    normalize_items,
)
from diffuplan_core.metrics import attach_metrics
from diffuplan_core.rollout import RolloutConfig, config_to_dict
from diffuplan_core.runtime import (
    detect_visible_gpu_ids,
    ensure_runtime_env,
    launch_retriever_async,
    probe_url,
    wait_for_retriever_ready,
)
from diffuplan_core.sampling import SamplingConfig
from diffuplan_core.search_client import SearchClient


DEFAULT_MODEL_PATH = os.environ.get("DIFFUPLAN_MODEL", str(DIFFUPLAN_ROOT.parent / "weights" / "diffuplan"))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run editable-plan LLaDA2-mini inference.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--question", action="append", help="Question text. Can be repeated.")
    source_group.add_argument("--input-file", help="Input .json/.jsonl/.txt file.")
    source_group.add_argument(
        "--dataset",
        action="append",
        help="Dataset name under --data-dir without .json suffix. Can be repeated.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DIFFUPLAN_ROOT.parent / "benchmarks"),
        help="Directory for --dataset JSON files.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k-sample", type=int, default=1)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--write-jsonl",
        default="",
        help="Optional flat records jsonl side output.",
    )

    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bootstrap torch.distributed workers across visible GPUs for dInfer sampling.",
    )
    parser.add_argument(
        "--nproc-per-node",
        default="auto",
        help="Worker count for torch.distributed.run; default uses visible GPU count.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--retriever-health-url", default="http://127.0.0.1:8000/docs")
    parser.add_argument("--retriever-topk", type=int, default=3)
    parser.add_argument("--fail-on-search-error", action="store_true")
    parser.add_argument(
        "--auto-launch-retriever",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch Diffuplan retriever if health check is not ready.",
    )
    parser.add_argument(
        "--retriever-root",
        default=str(DIFFUPLAN_ROOT),
        help="Directory containing launch_retriever.sh when --auto-launch-retriever is used.",
    )
    parser.add_argument("--retriever-log-dir", default=str(DIFFUPLAN_ROOT / "logs"))
    parser.add_argument("--retriever-startup-timeout", type=int, default=180)
    parser.add_argument("--retriever-gpu-ids", default="")
    parser.add_argument("--retriever-gpu-load-mode", default="batch")
    parser.add_argument("--no-retriever-gpu-shard", action="store_true")

    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-search-turns", type=int, default=5)
    parser.add_argument("--max-obs-tokens", type=int, default=500)
    parser.add_argument(
        "--recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append controller recovery observations after invalid actions or exhausted budgets.",
    )
    parser.add_argument("--invalid-action-retries", type=int, default=1)
    parser.add_argument("--answer-direct-retries", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument(
        "--sampler-backend",
        default="dinfer",
        choices=("dinfer", "transformers"),
        help="Backend for editable-plan action generation and plan edits.",
    )
    parser.add_argument(
        "--edit-sampler-backend",
        default=None,
        choices=("transformers", "dinfer"),
        help="Deprecated alias for --sampler-backend.",
    )
    parser.add_argument("--edit-dinfer-parallel-mode", default="dp", choices=("dp", "tp"))
    parser.add_argument("--edit-dinfer-max-batch-size", type=int, default=1)
    parser.add_argument("--edit-dinfer-prefilling-limit", type=int, default=128)
    parser.add_argument("--edit-dinfer-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-token-id", type=int, default=156895)
    parser.add_argument("--eos-token-id", type=int, default=156892)
    parser.add_argument("--expand-token", default="<|reserved_token_6|>")
    parser.add_argument("--no-init-expand-token", action="store_true")
    parser.add_argument("--print-limit", type=int, default=20)
    args = parser.parse_args(list(argv))
    if args.edit_sampler_backend is not None:
        args.sampler_backend = args.edit_sampler_backend
    return args


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
        dtype=dtype_map[args.dtype],
    )
    model.to(device)
    model.eval()

    expand_info = resolve_expand_token(tokenizer, preferred_text=args.expand_token)
    initialized = False
    if not args.no_init_expand_token:
        initialized = initialize_expand_token_mean(model, expand_info.token_id)

    meta = {
        "device": device,
        "dtype": args.dtype,
        "expand_token": {
            **asdict(expand_info),
            "initialized": bool(initialized),
        },
    }
    return model, tokenizer, meta


def build_rollout_config(args: argparse.Namespace, expand_token_id: int, expand_token_text: str) -> RolloutConfig:
    sampling = SamplingConfig(
        max_new_tokens=int(args.max_new_tokens),
        block_length=int(args.block_length),
        denoising_steps=int(args.denoising_steps),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        confidence_threshold=float(args.confidence_threshold),
        mask_token_id=int(args.mask_token_id),
        eos_token_id=int(args.eos_token_id),
        expand_token_id=int(expand_token_id),
        expand_token_text=expand_token_text,
    )
    return RolloutConfig(
        max_turns=int(args.max_turns),
        max_search_turns=int(args.max_search_turns),
        max_obs_tokens=int(args.max_obs_tokens),
        recovery_enabled=bool(args.recovery),
        invalid_action_retries=int(args.invalid_action_retries),
        answer_direct_retries=int(args.answer_direct_retries),
        sampling=sampling,
    )


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DIFFUPLAN_ROOT / "outputs" / f"llada2_mini_editable_plan_{timestamp}.json"


def maybe_launch_retriever(args: argparse.Namespace) -> dict[str, Any]:
    state_meta: dict[str, Any] = {
        "auto_launch": bool(args.auto_launch_retriever),
        "health_url": args.retriever_health_url,
        "url": args.retriever_url,
        "ready_before": probe_url(args.retriever_health_url),
        "launched": False,
        "reused_existing": False,
        "log_path": "",
        "visible_gpu_ids": detect_visible_gpu_ids(),
    }
    if state_meta["ready_before"] or not args.auto_launch_retriever:
        state_meta["ready_after"] = state_meta["ready_before"]
        return state_meta

    gpu_ids = [item for item in args.retriever_gpu_ids.split(",") if item.strip()]
    launch_state = launch_retriever_async(
        retriever_root=args.retriever_root,
        health_url=args.retriever_health_url,
        log_dir=args.retriever_log_dir,
        topk=int(args.retriever_topk),
        gpu_ids=gpu_ids or None,
        gpu_shard=not bool(args.no_retriever_gpu_shard),
        gpu_load_mode=args.retriever_gpu_load_mode,
    )
    state_meta["launched"] = launch_state.process is not None
    state_meta["reused_existing"] = launch_state.reused_existing
    state_meta["log_path"] = launch_state.log_path
    wait_for_retriever_ready(
        launch_state,
        health_url=args.retriever_health_url,
        timeout_sec=int(args.retriever_startup_timeout),
    )
    state_meta["ready_after"] = probe_url(args.retriever_health_url)
    return state_meta


def load_items_from_args(args: argparse.Namespace) -> list[InputItem]:
    rows = load_input_payload(
        question=args.question,
        input_file=args.input_file,
        dataset=args.dataset,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
    )
    return normalize_items(rows)


def is_distributed_worker() -> bool:
    return "RANK" in os.environ or "LOCAL_RANK" in os.environ


def bootstrap_distributed(args: argparse.Namespace) -> int:
    visible_gpu_ids = detect_visible_gpu_ids()
    if args.nproc_per_node == "auto":
        nproc = len(visible_gpu_ids)
    else:
        nproc = int(args.nproc_per_node)
    if nproc <= 1:
        return distributed_or_single_main(args)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    print(f"[Bootstrap] launching {nproc} distributed workers", flush=True)
    print(f"[Bootstrap] command: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, env=os.environ.copy(), check=False)
    return int(completed.returncode)


def init_distributed_if_needed() -> tuple[int, int, int, Any]:
    if not is_distributed_worker():
        return 0, 1, -1, None

    import torch
    import torch.distributed as dist
    from datetime import timedelta

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        kwargs = {
            "backend": "nccl",
            "init_method": "env://",
            "timeout": timedelta(seconds=int(os.environ.get("DIFFUPLAN_DIST_TIMEOUT_SEC", "1800"))),
        }
        try:
            dist.init_process_group(**kwargs, device_id=torch.device(f"cuda:{local_rank}"))
        except TypeError:
            dist.init_process_group(**kwargs)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank, dist


def distributed_barrier(dist: Any, local_rank: int) -> None:
    if dist is None:
        return
    try:
        dist.barrier(device_ids=[local_rank] if local_rank >= 0 else None)
    except TypeError:
        dist.barrier()


def contiguous_chunk(items: list[InputItem], world_size: int, rank: int) -> list[InputItem]:
    total = len(items)
    base = total // world_size
    rem = total % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return items[start:end]


def run_records(
    *,
    items: list[InputItem],
    model: Any,
    tokenizer: Any,
    search_client: SearchClient,
    rollout_config: RolloutConfig,
    action_dinfer_engine: Any = None,
    edit_dinfer_engine: Any = None,
    k_sample: int,
    print_limit: int,
) -> list[dict[str, Any]]:
    engine = EditablePlanInferenceEngine(
        model=model,
        tokenizer=tokenizer,
        search_client=search_client,
        config=rollout_config,
        action_dinfer_engine=action_dinfer_engine,
        edit_dinfer_engine=edit_dinfer_engine,
    )
    return engine.generate_rollouts(
        items=items,
        k_sample=k_sample,
        progress=True,
        print_limit=print_limit,
    )


def init_dinfer_engine(
    *,
    args: argparse.Namespace,
    dist: Any,
    world_size: int,
) -> tuple[Any, dict[str, Any]]:
    if dist is None or not dist.is_initialized():
        raise RuntimeError("--sampler-backend dinfer requires --distributed so torch.distributed is initialized.")

    from diffuplan_core.dinfer_runtime.dinfer_rollout_engine import (
        DInferRolloutEngine,
        DInferRuntimeConfig,
    )

    runtime_cfg = DInferRuntimeConfig(
        use_compile=bool(args.edit_dinfer_compile),
        deterministic_zero_temp=True,
        prefilling_limit=int(args.edit_dinfer_prefilling_limit),
        dist_timeout_sec=int(os.environ.get("DIFFUPLAN_DIST_TIMEOUT_SEC", "1800")),
    )
    engine = DInferRolloutEngine(
        model_path=str(args.model_path),
        retriever_url=args.retriever_url,
        topk_search=int(args.retriever_topk),
        max_batch_size=int(args.edit_dinfer_max_batch_size),
        length=max(int(args.max_new_tokens), 512),
        block_length=int(args.block_length),
        threshold=float(args.confidence_threshold),
        max_turns=1,
        max_obs_length=int(args.max_obs_tokens),
        runtime_cfg=runtime_cfg,
        parallel_mode=str(args.edit_dinfer_parallel_mode),
        unload_after_rollout=False,
        expert_map_path=str(DIFFUPLAN_ROOT),
    )
    engine.initialize()
    return engine, {
        "backend": "dinfer",
        "action_backend": "dinfer",
        "edit_backend": "dinfer",
        "parallel_mode": str(args.edit_dinfer_parallel_mode),
        "world_size": int(world_size),
        "compile": bool(args.edit_dinfer_compile),
        "max_batch_size": int(args.edit_dinfer_max_batch_size),
    }


def gather_rank_records(
    *,
    local_records: list[dict[str, Any]],
    dist: Any,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]] | None:
    if dist is None or world_size == 1:
        return local_records

    gather_dir_obj: list[str | None] = [None]
    if rank == 0:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        gather_dir = DIFFUPLAN_ROOT / "outputs" / f".gather_{run_tag}_{os.getpid()}"
        gather_dir.mkdir(parents=True, exist_ok=True)
        gather_dir_obj[0] = str(gather_dir)
    dist.broadcast_object_list(gather_dir_obj, src=0)
    gather_dir = Path(str(gather_dir_obj[0]))

    rank_path = gather_dir / f"rank_{rank}.json"
    tmp_path = gather_dir / f"rank_{rank}.json.tmp"
    with tmp_path.open("w") as handle:
        json.dump(local_records, handle, ensure_ascii=False)
    os.replace(tmp_path, rank_path)
    distributed_barrier(dist, int(os.environ.get("LOCAL_RANK", "-1")))

    if rank != 0:
        return None

    merged: list[dict[str, Any]] = []
    for src_rank in range(world_size):
        path = gather_dir / f"rank_{src_rank}.json"
        with path.open("r") as handle:
            merged.extend(json.load(handle))
    for src_rank in range(world_size):
        try:
            (gather_dir / f"rank_{src_rank}.json").unlink()
        except FileNotFoundError:
            pass
    try:
        gather_dir.rmdir()
    except OSError:
        pass
    return merged


def write_jsonl(path: str, records: list[dict[str, Any]]) -> Optional[Path]:
    if not path:
        return None
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def distributed_or_single_main(args: argparse.Namespace) -> int:
    ensure_runtime_env()
    rank, world_size, local_rank, dist = init_distributed_if_needed()
    items = load_items_from_args(args)
    local_items = contiguous_chunk(items, world_size, rank)

    retriever_meta: dict[str, Any] = {"rank0_only": True, "world_size": world_size}
    if rank == 0:
        retriever_meta = maybe_launch_retriever(args)
    if dist is not None:
        distributed_barrier(dist, local_rank)

    t_load = time.time()
    action_dinfer_engine = None
    edit_dinfer_engine = None
    if args.sampler_backend == "dinfer":
        model = None
        action_dinfer_engine, sampler_meta = init_dinfer_engine(
            args=args,
            dist=dist,
            world_size=world_size,
        )
        edit_dinfer_engine = action_dinfer_engine
        tokenizer = action_dinfer_engine.tokenizer
        expand_info = resolve_expand_token(tokenizer, preferred_text=args.expand_token)
        model_meta = {
            "backend": "dinfer",
            "device": f"cuda:{local_rank}" if local_rank >= 0 else "cuda",
            "dtype": args.dtype,
            "expand_token": {
                **asdict(expand_info),
                "initialized": False,
                "initialization_skipped": "dinfer_backend",
            },
        }
    else:
        model, tokenizer, model_meta = load_model_and_tokenizer(args)
        sampler_meta = {
            "backend": "transformers",
            "action_backend": "transformers",
            "edit_backend": "transformers",
            "world_size": int(world_size),
        }
    load_elapsed = time.time() - t_load

    expand_meta = model_meta["expand_token"]
    rollout_config = build_rollout_config(
        args,
        expand_token_id=int(expand_meta["token_id"]),
        expand_token_text=str(expand_meta["token_text"]),
    )
    search_client = SearchClient(
        url=args.retriever_url,
        topk=int(args.retriever_topk),
        fail_on_error=bool(args.fail_on_search_error),
    )
    records = run_records(
        items=local_items,
        model=model,
        tokenizer=tokenizer,
        search_client=search_client,
        rollout_config=rollout_config,
        action_dinfer_engine=action_dinfer_engine,
        edit_dinfer_engine=edit_dinfer_engine,
        k_sample=int(args.k_sample),
        print_limit=int(args.print_limit),
    )
    merged_records = gather_rank_records(
        local_records=records,
        dist=dist,
        rank=rank,
        world_size=world_size,
    )
    if rank != 0:
        if dist is not None:
            dist.destroy_process_group()
        return 0

    records = merged_records or []
    metrics_summary = attach_metrics(records)

    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = write_jsonl(args.write_jsonl, records)
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "model_path": args.model_path,
            "model_load_elapsed_s": load_elapsed,
            "model": model_meta,
            "input_count": len(items),
            "k_sample": int(args.k_sample),
            "dataset_names": sorted({item.dataset_name for item in items if item.dataset_name}),
            "world_size": int(world_size),
            "retriever_url": args.retriever_url,
            "retriever_topk": int(args.retriever_topk),
            "retriever": retriever_meta,
            "rollout_config": config_to_dict(rollout_config),
            "sampler": sampler_meta,
            "edit_sampler": sampler_meta,
        },
        "metrics": metrics_summary,
        "results": records,
    }
    with output_path.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    overall = metrics_summary.get("overall", {})
    print(
        "[Metrics][overall] "
        f"count={int(overall.get('count', 0) or 0)} "
        f"EM={float(overall.get('mean_em', 0.0) or 0.0):.4f} "
        f"SubEM={float(overall.get('mean_subem', 0.0) or 0.0):.4f} "
        f"F1={float(overall.get('mean_f1', 0.0) or 0.0):.4f} "
        f"FmtErr={float(overall.get('format_error_frac', 0.0) or 0.0):.4f}",
        flush=True,
    )
    if jsonl_path is not None:
        print(f"[Result] jsonl saved={jsonl_path}", flush=True)
    print(f"[Result] rows={len(records)} saved={output_path}", flush=True)
    if dist is not None:
        dist.destroy_process_group()
    return 0


def main(argv: Sequence[str]) -> int:
    ensure_runtime_env()
    args = parse_args(argv)
    if args.distributed and args.device != "cpu" and not is_distributed_worker():
        return bootstrap_distributed(args)
    return distributed_or_single_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
