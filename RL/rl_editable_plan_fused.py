#!/usr/bin/env python
"""Distributed Agentic TraceRL training with dInfer rollouts and a frozen SFT anchor.

Each FSDP rank generates traced search episodes, reconstructs Action-Mode rows,
applies PPO updates, and synchronizes the rollout model."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Resolve the runtime and learner modules from this checkout.
_RL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RL_DIR.parent
_DIFFUPLAN_ROOT = _REPO_ROOT / "Diffuplan"
for _root in (str(_DIFFUPLAN_ROOT), str(_REPO_ROOT), str(_RL_DIR)):
    if _root in sys.path:
        sys.path.remove(_root)
    sys.path.insert(0, _root)

# Match TraceRL's launcher behavior: keep the terminal readable by letting only
# rank0 write to stdout/stderr; all other local ranks go to logs/rank_i.out.
_RANK_LOG_HANDLE = None
if os.environ.get("RL_REDIRECT_NON_MAIN_RANKS", "1").strip().lower() not in {"0", "false", "no", "off"}:
    try:
        _LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        _LOCAL_RANK = 0
    if _LOCAL_RANK != 0:
        (_RL_DIR / "logs").mkdir(parents=True, exist_ok=True)
        _RANK_LOG_HANDLE = open(_RL_DIR / "logs" / f"rank_{_LOCAL_RANK}.out", "w", buffering=1)
        sys.stdout = _RANK_LOG_HANDLE
        sys.stderr = _RANK_LOG_HANDLE
        try:
            os.dup2(sys.stdout.fileno(), 1)
            os.dup2(sys.stderr.fileno(), 2)
        except Exception:
            pass

from diffuplan_core.rollout import RolloutConfig  # noqa: E402
from diffuplan_core.metrics import attach_metrics  # noqa: E402
from diffuplan_core.sampling import SamplingConfig  # noqa: E402
from diffuplan_core.search_client import SearchClient  # noqa: E402
from diffuplan_core.runtime import probe_url, wait_for_retriever_ready, RetrieverState  # noqa: E402

import losses as L  # noqa: E402
import reward_format_guard as RFG  # noqa: E402
import trace_to_rows as TTR  # noqa: E402
from config_utils import get_config  # noqa: E402
from rollout_with_trace import run_traced_episode, run_traced_episodes  # noqa: E402


# --------------------------------------------------------------------------- utils
def _is_main(accelerator: Accelerator) -> bool:
    return accelerator.is_main_process


def _log(accelerator: Accelerator, msg: str) -> None:
    if _is_main(accelerator):
        print(msg, flush=True)


def _banner(accelerator: Accelerator, step: int, total: int, phase: str, extra: str = "") -> None:
    """Print a training phase on rank 0."""
    if _is_main(accelerator):
        bar = "=" * 18
        tail = f" | {extra}" if extra else ""
        print(f"\n{bar} [step {step}/{total}] {phase}{tail} {bar}", flush=True)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _progress_due(done: int, total: int, last_log_time: float, every: int, interval_sec: float) -> bool:
    if done <= 0:
        return False
    if done == 1 or done >= total:
        return True
    if every > 0 and done % every == 0:
        return True
    if interval_sec > 0 and time.time() - last_log_time >= interval_sec:
        return True
    return False


def _progress_rank_enabled(accelerator: Accelerator, rank: int, *, env_name: str, default: str) -> bool:
    if os.environ.get("RL_PROGRESS", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    mode = os.environ.get(env_name, os.environ.get("RL_PROGRESS_RANKS", default)).strip().lower()
    if mode in {"all", "*"}:
        return True
    if mode in {"main", "rank0", "0"}:
        return _is_main(accelerator)
    allowed = {part.strip().removeprefix("rank") for part in mode.split(",") if part.strip()}
    return str(rank) in allowed


def _progress_print(
    accelerator: Accelerator,
    rank: int,
    phase: str,
    msg: str,
    *,
    env_name: str,
    default_ranks: str,
) -> None:
    if not _progress_rank_enabled(accelerator, rank, env_name=env_name, default=default_ranks):
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}][{phase}][rank{rank}] {msg}", flush=True)


def _progress_bar(
    iterable,
    accelerator: Accelerator,
    rank: int,
    *,
    desc: str,
    total: Optional[int] = None,
    leave: bool = False,
    default_ranks: str = "all",
):
    enabled = _progress_rank_enabled(accelerator, rank, env_name="RL_TQDM_RANKS", default=default_ranks)
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        disable=not enabled,
        leave=leave,
        dynamic_ncols=True,
        mininterval=1.0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]{postfix}",
    )


@contextlib.contextmanager
def _suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _broadcast_object(obj: Any, src: int = 0) -> Any:
    holder = [obj]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def _all_gather_object(obj: Any, world_size: int) -> List[Any]:
    gathered: List[Any] = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def _contiguous_chunk(n: int, world_size: int, rank: int) -> tuple[int, int]:
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def _global_max_count(local_count: int, world_size: int) -> int:
    """All-reduce the max local item count so every rank runs the same #collectives (FSDP lockstep)."""
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return local_count
    t = torch.tensor([int(local_count)], dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def _global_sum_float(local_value: float, world_size: int) -> float:
    """All-reduce a float sum across ranks (used to size the GLOBAL edit accumulation window)."""
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return float(local_value)
    t = torch.tensor([float(local_value)], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _pad_with_dummies(items: list, target: int, make_dummy) -> list:
    """Pad `items` up to `target` with zero-loss-weight dummies (zero gradient, keeps NCCL aligned)."""
    if len(items) >= target:
        return items
    if items:
        seed = items[0]
    else:
        seed = make_dummy()
    out = list(items)
    while len(out) < target:
        out.append(make_dummy(seed))
    return out


def _checkpoint_has_qk_norm(model_path: str) -> bool:
    """True if the checkpoint contains query/key layernorm weights (=> use_qk_norm)."""
    idx = Path(model_path) / "model.safetensors.index.json"
    try:
        if idx.is_file():
            wmap = json.loads(idx.read_text()).get("weight_map", {})
            return any(("query_layernorm" in k or "key_layernorm" in k) for k in wmap)
    except Exception:
        pass
    return False


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
    return path


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return path


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to decode rollout shard {path}:{line_no}: {exc}") from exc
    return records


def _gather_episode_shards_via_files(
    *,
    local_episodes: List[Dict[str, Any]],
    project_dir: Path,
    step: int,
    gather_tag: str,
    world_size: int,
    rank: int,
) -> List[Dict[str, Any]]:
    """Gather large rollout payloads without dist.all_gather_object's GPU tensor path."""
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return list(local_episodes)

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(gather_tag))
    gather_dir = project_dir / "tmp" / "rollout_gather" / f"{safe_tag}_step_{int(step):04d}"
    gather_dir.mkdir(parents=True, exist_ok=True)
    shard_path = gather_dir / f"rank_{int(rank):04d}.jsonl"
    tmp_path = shard_path.with_suffix(".jsonl.tmp")
    _write_jsonl(tmp_path, local_episodes)
    tmp_path.replace(shard_path)

    dist.barrier()
    episodes: List[Dict[str, Any]] = []
    missing: List[str] = []
    for shard_rank in range(world_size):
        path = gather_dir / f"rank_{shard_rank:04d}.jsonl"
        if not path.is_file():
            missing.append(str(path))
            continue
        episodes.extend(_read_jsonl(path))
    if missing:
        raise RuntimeError("Missing rollout shard(s): " + ", ".join(missing))
    dist.barrier()
    episodes.sort(key=lambda ep: (int(ep["input_index"]), int(ep["sample_index"])))
    return episodes


def _project_dir(project_name: str) -> Path:
    return Path(project_name).expanduser().resolve()


def _prepare_project_dir(accelerator: Accelerator, project_name: str, *, start_from_scratch: bool) -> Path:
    project_dir = _project_dir(project_name)
    if accelerator.is_main_process:
        project_dir.mkdir(parents=True, exist_ok=True)
        if start_from_scratch:
            removed: List[str] = []
            for child_name in ("ckpt", "traces", "eval"):
                child = project_dir / child_name
                if child.exists():
                    shutil.rmtree(child)
                    removed.append(child_name)
            if removed:
                print(f"[Setup] start_from_scratch=True -> removed {', '.join(removed)} under {project_dir}", flush=True)
    accelerator.wait_for_everyone()
    return project_dir


def _checkpoint_registry_path(project_dir: Path) -> Path:
    return project_dir / "ckpt" / "best_eval_checkpoints.json"


def _sort_eval_checkpoint_entries(
    entries: List[Dict[str, Any]],
    *,
    metric_mode: str,
) -> List[Dict[str, Any]]:
    mode = str(metric_mode).strip().lower()
    if mode == "max":
        return sorted(
            entries,
            key=lambda item: (float(item.get("score", float("-inf"))), int(item.get("step", -1))),
            reverse=True,
        )
    if mode == "min":
        return sorted(
            entries,
            key=lambda item: (float(item.get("score", float("inf"))), -int(item.get("step", -1))),
        )
    raise ValueError(f"Unsupported checkpoint metric mode: {metric_mode!r}")


def _topk_eval_checkpoint_entries(
    entries: List[Dict[str, Any]],
    *,
    top_k: int,
    metric_mode: str,
) -> List[Dict[str, Any]]:
    top_k = max(0, int(top_k))
    if top_k <= 0:
        return []
    return _sort_eval_checkpoint_entries(entries, metric_mode=metric_mode)[:top_k]


def _eval_checkpoint_step_is_eligible(step: int, *, min_step: int) -> bool:
    return int(step) >= max(1, int(min_step))


def _build_eval_checkpoint_entry(
    step: int,
    metrics_summary: Dict[str, Any],
    *,
    metric_name: str,
) -> Optional[Dict[str, Any]]:
    overall = dict(metrics_summary.get("overall", {}) or {})
    if metric_name not in overall:
        return None
    try:
        score = float(overall.get(metric_name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    return {
        "step": int(step),
        "score": score,
        "overall": overall,
        "path": f"step_{int(step)}",
    }


def _torch_dtype_config_name(dtype: Optional[torch.dtype]) -> Optional[str]:
    dtype = _coerce_torch_dtype(dtype)
    mapping = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    return mapping.get(dtype)


def _coerce_torch_dtype(value: Any) -> Optional[torch.dtype]:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        mapping = {
            "float16": torch.float16,
            "torch.float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "torch.bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "torch.float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float64": torch.float64,
            "torch.float64": torch.float64,
            "fp64": torch.float64,
            "double": torch.float64,
        }
        return mapping.get(value.strip().lower())
    return None


def _iter_named_floating_tensors(module: Any):
    if module is None:
        return
    for iterator_name in ("named_parameters", "named_buffers"):
        iterator = getattr(module, iterator_name, None)
        if iterator is None:
            continue
        try:
            for name, tensor in iterator():
                if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
                    yield str(name), tensor
        except Exception:
            continue


def _floating_dtype_map(module: Any) -> Dict[str, torch.dtype]:
    return {name: tensor.dtype for name, tensor in _iter_named_floating_tensors(module)}


def _infer_dominant_floating_dtype(module: Any) -> Optional[torch.dtype]:
    totals: Dict[torch.dtype, int] = {}
    for _, tensor in _iter_named_floating_tensors(module):
        totals[tensor.dtype] = totals.get(tensor.dtype, 0) + int(tensor.numel())
    if not totals:
        return None
    return max(totals.items(), key=lambda item: item[1])[0]


def _infer_config_dtype(config: Any) -> Optional[torch.dtype]:
    # Prefer torch_dtype over dtype because some older saved configs carry
    # dtype=float32 while torch_dtype=bfloat16.
    for attr in ("torch_dtype", "dtype"):
        dtype = _coerce_torch_dtype(getattr(config, attr, None))
        if dtype is not None:
            return dtype
    return None


def _infer_eval_runtime_dtype(eval_engine: Any) -> Optional[torch.dtype]:
    runtime_model = getattr(eval_engine, "model", eval_engine)
    live_dtype = _infer_dominant_floating_dtype(runtime_model)
    if live_dtype is not None:
        return live_dtype
    runtime_config = getattr(eval_engine, "_model_config", None)
    config_dtype = _infer_config_dtype(runtime_config)
    if config_dtype is not None:
        return config_dtype
    model_config = getattr(runtime_model, "config", None)
    return _infer_config_dtype(model_config)


def _cast_state_dict_for_eval_runtime(
    state_dict: Dict[str, torch.Tensor],
    *,
    eval_engine: Any,
    fallback_dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    runtime_model = getattr(eval_engine, "model", eval_engine)
    dtype_by_name = _floating_dtype_map(runtime_model)
    for key, tensor in list(state_dict.items()):
        if not (torch.is_tensor(tensor) and torch.is_floating_point(tensor)):
            continue
        target_dtype = dtype_by_name.get(str(key), fallback_dtype)
        if target_dtype is not None and tensor.dtype != target_dtype:
            state_dict[key] = tensor.detach().to(dtype=target_dtype)
    return state_dict


def _patch_saved_config_dtype(save_path: Path, dtype: Optional[torch.dtype]) -> None:
    dtype_name = _torch_dtype_config_name(dtype)
    if not dtype_name:
        return
    config_path = save_path / "config.json"
    if not config_path.exists():
        return
    with config_path.open("r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return
    payload["dtype"] = dtype_name
    payload["torch_dtype"] = dtype_name
    with config_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _load_checkpoint_registry(
    project_dir: Path,
    *,
    metric_name: str,
    metric_mode: str,
    top_k: int,
) -> Dict[str, Any]:
    path = _checkpoint_registry_path(project_dir)
    default = {
        "metric_name": str(metric_name),
        "metric_mode": str(metric_mode),
        "top_k": int(top_k),
        "checkpoints": [],
    }
    if not path.exists():
        return default
    try:
        with path.open("r") as handle:
            payload = json.load(handle)
    except Exception:
        return default
    if not isinstance(payload, dict):
        return default
    payload.setdefault("metric_name", str(metric_name))
    payload.setdefault("metric_mode", str(metric_mode))
    payload.setdefault("top_k", int(top_k))
    payload.setdefault("checkpoints", [])
    if not isinstance(payload.get("checkpoints"), list):
        payload["checkpoints"] = []
    return payload


def _broadcast_main_process_value(accelerator: Accelerator, value: Any) -> Any:
    payload = [value if accelerator.is_main_process else None]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _load_named_dataset(name: str, *, shuffle: bool = False, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    path = _RL_DIR / "data" / f"{name}.json"
    with open(path, "r") as f:
        data = json.load(f)
    if shuffle:
        rng = random.Random(0 if seed is None else int(seed))
        rng.shuffle(data)
    return data


def load_training_data(config) -> List[Dict[str, Any]]:
    return _load_named_dataset(
        str(config.dataset.train_dataset),
        shuffle=True,
        seed=int(config.training.seed),
    )


_EVAL_DATASET_FILES = {
    "hotpotqa": ("hotpotqa", _RL_DIR / "data" / "HotpotQA_val.json"),
    "HotpotQA_val": ("hotpotqa", _RL_DIR / "data" / "HotpotQA_val.json"),
    "2wiki": ("2wiki", _RL_DIR / "data" / "2wiki_val.json"),
    "2wiki_val": ("2wiki", _RL_DIR / "data" / "2wiki_val.json"),
}


def _load_eval_named_dataset(name: str) -> tuple[str, List[Dict[str, Any]]]:
    key = str(name).strip()
    if key in _EVAL_DATASET_FILES:
        canonical_name, path = _EVAL_DATASET_FILES[key]
    else:
        canonical_name = key
        path = _RL_DIR / "data" / f"{key}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Eval dataset {key!r} not found at {path}")
    with path.open("r") as f:
        data = json.load(f)
    return canonical_name, data


def load_eval_data(config) -> List[Dict[str, Any]]:
    eval_cfg = config.get("evaluation", {})
    # Prefer the multi-dataset list (eval_datasets); fall back to the single eval_dataset.
    names = eval_cfg.get("eval_datasets", None)
    if names:
        dataset_names = [str(n).strip() for n in names if str(n).strip()]
    else:
        single = str(eval_cfg.get("eval_dataset", "") or "").strip()
        dataset_names = [single] if single else []
    if not dataset_names:
        return []
    max_samples = int(eval_cfg.get("max_eval_samples", 0) or 0)
    combined: List[Dict[str, Any]] = []
    for name in dataset_names:
        canonical_name, data = _load_eval_named_dataset(name)
        if max_samples > 0:
            data = data[:max_samples]
        for item in data:
            tagged = dict(item)
            tagged["source_dataset"] = canonical_name
            combined.append(tagged)
    return combined


def _record_dinfer_residency(*, project_dir: Path, stage: str, step: int,
                             rank: int, engine: Any, learner: Any) -> None:
    """Record dInfer GPU residency and memory usage."""
    if os.environ.get("RL_RESIDENCY_AUDIT", "0").lower() not in {"1", "true", "yes"}:
        return
    params = list(engine.model.named_parameters())
    if not params or any(t.device.type != "cuda" for _, t in params):
        raise RuntimeError("Resident dInfer has a missing or non-CUDA parameter")
    if not engine.weights_loaded or engine.parallel_mode != "dp":
        raise RuntimeError("Resident dInfer must have loaded DP weights")
    device_bytes: Dict[str, int] = {}
    for name, tensor in params:
        device = str(tensor.device)
        device_bytes[device] = device_bytes.get(device, 0) + tensor.numel() * tensor.element_size()
    free, total = torch.cuda.mem_get_info(engine.device)
    record = {
        "stage": stage, "step": step, "rank": rank,
        "parameter_bytes_by_device": device_bytes,
        "runtime_ids": {k: id(getattr(engine, k)) for k in ("model", "model_runner", "decoder", "dllm")},
        "learner_sharding": str(getattr(learner, "sharding_strategy", "unknown")),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(engine.device),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(engine.device),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(engine.device),
        "device_used_bytes": total - free, "device_total_bytes": total,
        "sync_rebound_storage": bool(getattr(engine, "_last_sync_used_rebind", False)),
    }
    directory = project_dir / "residency"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"rank_{rank}.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"[resident][rank{rank}] {stage} step={step} "
          f"device_used={record['device_used_bytes'] / 2**30:.2f}GiB "
          f"peak_allocated={record['cuda_peak_allocated_bytes'] / 2**30:.2f}GiB "
          f"rebind={record['sync_rebound_storage']}", flush=True)


def _get_train_state_dict_for_runtime_sync(accelerator: Accelerator, model, *, rank0_only=False) -> Dict[str, torch.Tensor]:
    """Full (unsharded) state dict for syncing into the dInfer engine. Mirrors TraceRL."""
    if getattr(accelerator, "is_fsdp2", False):
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=False, cpu_offload=True)
        return get_model_state_dict(model, options=options)
    if accelerator.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    return accelerator.unwrap_model(model).state_dict()


def build_rollout_config(config, *, sampling_overrides: Optional[Dict[str, Any]] = None) -> RolloutConfig:
    r = config.rollout
    s = config.search
    sampling_overrides = dict(sampling_overrides or {})
    sampling = SamplingConfig(
        max_new_tokens=int(r.get("max_new_tokens", 256)),
        block_length=int(r.get("block_size", 32)),
        denoising_steps=int(r.get("denoising_steps_per_block", 32)),
        temperature=float(sampling_overrides.get("temperature", r.get("temperature", 0.8))),
        top_p=float(sampling_overrides.get("top_p", r.get("top_p", 0.95))),
        top_k=int(sampling_overrides.get("top_k", r.get("top_k", 0))),
        confidence_threshold=float(sampling_overrides.get("confidence_threshold", r.get("confidence_threshold", 0.9))),
        edit_transfer_tokens=int(config.get("edit", {}).get("edit_transfer_tokens", 1)),
        max_expand_per_edit=int(config.get("edit", {}).get("max_expand_per_edit", 32)),
    )
    return RolloutConfig(
        max_turns=int(s.get("max_turns", 12)),
        max_search_turns=int(s.get("max_retrieval_turns", 5)),
        max_obs_tokens=int(s.get("max_obs_length", 500)),
        recovery_enabled=True,
        sampling=sampling,
    )


def build_eval_rollout_config(config) -> RolloutConfig:
    eval_cfg = config.get("evaluation", {})
    sampling_overrides = {
        "temperature": float(eval_cfg.get("temperature", 0.0)),
        "top_p": float(eval_cfg.get("top_p", 1.0)),
        "top_k": int(eval_cfg.get("top_k", 0)),
    }
    if "confidence_threshold" in eval_cfg:
        sampling_overrides["confidence_threshold"] = float(eval_cfg.get("confidence_threshold"))
    return build_rollout_config(config, sampling_overrides=sampling_overrides)


# --------------------------------------------------------------------------- rollout
def run_rollouts_distributed(
    accelerator: Accelerator,
    *,
    prompts: List[Dict[str, Any]],
    k_sample: int,
    dinfer_engine: Any,
    tokenizer: Any,
    search_client: SearchClient,
    rollout_config: RolloutConfig,
    world_size: int,
    rank: int,
    project_dir: Path,
    step: int,
    gather_tag: str = "train",
) -> List[Dict[str, Any]]:
    """Each rank rolls out a contiguous shard of (prompt x k) tasks; gather to all ranks."""
    # Expand to (prompt_idx, sample_idx) tasks, sharded contiguously.
    tasks = [(pi, si) for pi in range(len(prompts)) for si in range(k_sample)]
    start, end = _contiguous_chunk(len(tasks), world_size, rank)
    local_tasks = (tasks[rank::world_size] if getattr(dinfer_engine, "interleave_tasks", False)
                   else tasks[start:end])
    local_total = len(local_tasks)
    if getattr(dinfer_engine, "interleave_tasks", False) and int(k_sample) > 1:
        # Interleaving puts samples of the same prompt on different DP ranks.
        # Give ranks independent random streams; identical initial CUDA seeds
        # would otherwise silently duplicate K samples across GPUs.
        rollout_seed = int(getattr(dinfer_engine, "rollout_seed", 10086)) + 1000003 * rank + 8191 * step
        with torch.cuda.device(dinfer_engine.device):
            torch.cuda.manual_seed(rollout_seed)
        _progress_print(accelerator, rank, "rollout-rng", f"seed={rollout_seed}",
                        env_name="RL_ROLLOUT_PROGRESS_RANKS", default_ranks="all")
    rollout_interval = _env_float("RL_ROLLOUT_PROGRESS_INTERVAL_SEC", _env_float("RL_PROGRESS_INTERVAL_SEC", 15.0))
    rollout_every = _env_int("RL_ROLLOUT_PROGRESS_EVERY", _env_int("RL_PROGRESS_EVERY", 8))
    progress_t0 = time.time()
    last_progress_log = progress_t0
    _progress_print(
        accelerator,
        rank,
        "rollout",
        f"start local={local_total}/{len(tasks)} shard={start}:{end} k={k_sample}",
        env_name="RL_ROLLOUT_PROGRESS_RANKS",
        default_ranks="all",
    )
    local_episodes: List[Dict[str, Any]] = []
    batch_size = int(getattr(dinfer_engine, "editable_batch_size", 1))
    if batch_size > 1:
        specs = [dict(question=str(prompts[pi].get("question", "")),
                      reference=prompts[pi].get("answer", prompts[pi].get("target", prompts[pi].get("ground_truth_answer", []))),
                      input_index=pi, sample_index=si,
                      dataset_name=str(prompts[pi].get("source_dataset", prompts[pi].get("dataset_name", ""))))
                 for pi, si in local_tasks]
        completed_count = 0
        def on_done(index, episode):
            nonlocal completed_count, last_progress_log
            completed_count += 1
            if _progress_due(completed_count, local_total, last_progress_log, rollout_every, rollout_interval):
                _progress_print(accelerator, rank, "rollout",
                    f"local={completed_count}/{local_total} batch={batch_size} elapsed={_fmt_seconds(time.time() - progress_t0)}",
                    env_name="RL_ROLLOUT_PROGRESS_RANKS", default_ranks="all")
                last_progress_log = time.time()
        local_episodes = run_traced_episodes(tasks=specs, tokenizer=tokenizer,
            search_client=search_client, config=rollout_config, dinfer_engine=dinfer_engine,
            batch_size=batch_size, on_done=on_done)
    else:
        for local_done, (pi, si) in enumerate(local_tasks, start=1):
            item = prompts[pi]
            ep = run_traced_episode(
                model=None,
                tokenizer=tokenizer,
                question=str(item.get("question", "")),
                reference=item.get("answer", item.get("target", item.get("ground_truth_answer", []))),
                search_client=search_client,
                config=rollout_config,
                action_dinfer_engine=dinfer_engine,
                edit_dinfer_engine=dinfer_engine,
                input_index=int(pi),
                sample_index=int(si),
                dataset_name=str(item.get("source_dataset", item.get("dataset_name", ""))),
            )
            local_episodes.append(ep)
            if _progress_due(local_done, local_total, last_progress_log, rollout_every, rollout_interval):
                approx_global_done = min(len(tasks), local_done * max(world_size, 1))
                _progress_print(
                    accelerator,
                    rank,
                    "rollout",
                    "local={}/{} approx_global={}/{} elapsed={} last=prompt{} sample{} "
                    "stop={} search={} upd={} replan={}".format(
                        local_done,
                        local_total,
                        approx_global_done,
                        len(tasks),
                        _fmt_seconds(time.time() - progress_t0),
                        pi,
                        si,
                        ep.get("stop_reason", ""),
                        int(ep.get("num_search", ep.get("retrieval_used", 0)) or 0),
                        int(ep.get("num_update", 0) or 0),
                        int(ep.get("num_replan", 0) or 0),
                    ),
                    env_name="RL_ROLLOUT_PROGRESS_RANKS",
                    default_ranks="all",
                )
                last_progress_log = time.time()
    _progress_print(
        accelerator,
        rank,
        "rollout",
        f"done local={local_total}/{len(tasks)} elapsed={_fmt_seconds(time.time() - progress_t0)}",
        env_name="RL_ROLLOUT_PROGRESS_RANKS",
        default_ranks="all",
    )

    return _gather_episode_shards_via_files(
        local_episodes=local_episodes,
        project_dir=project_dir,
        step=step,
        gather_tag=gather_tag,
        world_size=world_size,
        rank=rank,
    )


# --------------------------------------------------------------------------- row build
def build_all_rows(
    episodes: List[Dict[str, Any]],
    *,
    config,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    max_segments = int(config.training.get("max_train_segments_per_episode", 0) or 0)
    action_rows: List[Dict[str, Any]] = []
    for ep in episodes:
        lw = float(ep.get("loss_weight", 1.0))
        rew = float(ep.get("reward", 0.0))
        if lw <= 0.0:
            continue  # zero-variance group carries no signal
        action_rows.extend(
            TTR.build_action_rows(ep, reward=rew, loss_weight=lw, max_segments=max_segments, rng=rng)
        )
    return action_rows


def _save_training_traces(
    accelerator: Accelerator,
    *,
    project_dir: Path,
    step: int,
    episodes: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    trace_dir = project_dir / "traces"
    trace_prefix = f"train_step_{step:04d}"
    if accelerator.is_main_process:
        jsonl_path = _write_jsonl(trace_dir / f"{trace_prefix}.jsonl", episodes)
        summary_path = _write_json(
            trace_dir / f"{trace_prefix}.summary.json",
            {
                "meta": {
                    "step": int(step),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_episodes": int(len(episodes)),
                },
                "summary": summary,
            },
        )
        print(f"[trace] training traces -> {jsonl_path}", flush=True)
        print(f"[trace] training summary -> {summary_path}", flush=True)
    accelerator.wait_for_everyone()


def _dataset_key(name: Any) -> str:
    value = str(name or "").strip()
    mapping = {
        "HotpotQA_val": "hotpotqa",
        "hotpotqa": "hotpotqa",
        "2wiki_val": "2wiki",
        "2wiki": "2wiki",
    }
    return mapping.get(value, value)


def _summarize_metric_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def avg(key: str) -> float:
        return float(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows)) if rows else 0.0

    hard_count = sum(1 for row in rows if row.get("format_class") == "hard")
    soft_count = sum(1 for row in rows if row.get("format_class") == "soft")
    return {
        "count": float(len(rows)),
        "mean_em": avg("em"),
        "mean_subem": avg("subem"),
        "mean_f1": avg("f1"),
        "avg_retrieval_used": avg("retrieval_used"),
        "avg_response_length": avg("response_length"),
        "avg_generated_tokens_model_only": avg("generated_tokens_model_only"),
        "format_error_frac": ((hard_count + soft_count) / len(rows)) if rows else 0.0,
        "hard_format_count": float(hard_count),
        "soft_format_count": float(soft_count),
    }


def _run_pre_step_eval(
    accelerator: Accelerator,
    *,
    project_dir: Path,
    step: int,
    total_steps: int,
    eval_items: List[Dict[str, Any]],
    eval_k_sample: int,
    dinfer_engine: Any,
    tokenizer: Any,
    search_client: SearchClient,
    rollout_config: RolloutConfig,
    world_size: int,
    rank: int,
) -> Optional[Dict[str, Any]]:
    if not eval_items:
        return None

    _banner(
        accelerator,
        step,
        total_steps,
        "PRE-STEP EVAL",
        f"{len(eval_items)} prompts x {eval_k_sample} samples across {world_size} ranks",
    )
    t0 = time.time()
    records = run_rollouts_distributed(
        accelerator,
        prompts=eval_items,
        k_sample=eval_k_sample,
        dinfer_engine=dinfer_engine,
        tokenizer=tokenizer,
        search_client=search_client,
        rollout_config=rollout_config,
        world_size=world_size,
        rank=rank,
        project_dir=project_dir,
        step=step,
        gather_tag="eval",
    )
    accelerator.wait_for_everyone()
    metrics_summary = attach_metrics(records)
    overall = metrics_summary.get("overall", {})
    # Checkpoint score: mean F1 across HotpotQA and 2Wiki.
    avg_targets: List[float] = []
    per_dataset_log: List[str] = []
    for canonical_name in ("hotpotqa", "2wiki"):
        ds_name = next((name for name in metrics_summary if _dataset_key(name) == canonical_name), canonical_name)
        ds_metrics = metrics_summary.get(ds_name)
        if isinstance(ds_metrics, dict) and ds_metrics.get("mean_f1") is not None:
            f1 = float(ds_metrics.get("mean_f1", 0.0) or 0.0)
            avg_targets.append(f1)
            per_dataset_log.append(f"{ds_name}:f1={f1:.4f}(n={int(ds_metrics.get('count', 0) or 0)})")
    if isinstance(overall, dict) and avg_targets:
        overall["hotpot_2wiki_avg_f1"] = float(sum(avg_targets) / len(avg_targets))
    target_rows = [row for row in records if _dataset_key(row.get("dataset_name")) in {"hotpotqa", "2wiki"}]
    target_overall = _summarize_metric_rows(target_rows)
    if isinstance(overall, dict):
        target_overall["hotpot_2wiki_avg_f1"] = float(overall.get("hotpot_2wiki_avg_f1", 0.0) or 0.0)
    metrics_summary["hotpot_2wiki"] = target_overall
    _log(
        accelerator,
        "  >> eval done in {} | count={} EM={:.4f} SubEM={:.4f} F1={:.4f} FmtErr={:.4f} | avg_f1={:.4f} [{}]".format(
            _fmt_seconds(time.time() - t0),
            int(target_overall.get("count", 0) or 0),
            float(target_overall.get("mean_em", 0.0) or 0.0),
            float(target_overall.get("mean_subem", 0.0) or 0.0),
            float(target_overall.get("mean_f1", 0.0) or 0.0),
            float(target_overall.get("format_error_frac", 0.0) or 0.0),
            float(target_overall.get("hotpot_2wiki_avg_f1", 0.0) or 0.0),
            " ".join(per_dataset_log),
        ),
    )
    if accelerator.is_main_process:
        eval_dir = project_dir / "eval"
        prefix = f"pre_step_{step:04d}"
        jsonl_path = _write_jsonl(eval_dir / f"{prefix}.jsonl", records)
        payload_path = _write_json(
            eval_dir / f"{prefix}.json",
            {
                "meta": {
                    "step": int(step),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_prompts": int(len(eval_items)),
                    "k_sample": int(eval_k_sample),
                    "num_records": int(len(records)),
                },
                "metrics": metrics_summary,
                "results": records,
            },
        )
        print(f"[eval] records -> {jsonl_path}", flush=True)
        print(f"[eval] payload -> {payload_path}", flush=True)
    accelerator.wait_for_everyone()
    return metrics_summary


def _maybe_save_best_eval_checkpoint(
    accelerator: Accelerator,
    *,
    model: Any,
    tokenizer: Any,
    eval_engine: Any,
    project_dir: Path,
    step: int,
    source_model_path: str,
    metrics_summary: Optional[Dict[str, Any]],
    metric_name: str,
    metric_mode: str,
    top_k: int,
    min_step: int,
) -> bool:
    top_k = max(0, int(top_k))
    if top_k <= 0 or not metrics_summary:
        accelerator.wait_for_everyone()
        return False
    if not _eval_checkpoint_step_is_eligible(step, min_step=min_step):
        if accelerator.is_main_process:
            print(
                f"[save] skipped checkpoint at step {step}: before checkpoint_min_step={max(1, int(min_step))}",
                flush=True,
            )
        accelerator.wait_for_everyone()
        return False

    candidate: Optional[Dict[str, Any]] = None
    should_save = False
    kept_entries: List[Dict[str, Any]] = []
    removed_entries: List[Dict[str, Any]] = []
    if accelerator.is_main_process:
        registry = _load_checkpoint_registry(
            project_dir,
            metric_name=metric_name,
            metric_mode=metric_mode,
            top_k=top_k,
        )
        existing_all = [
            dict(item)
            for item in list(registry.get("checkpoints", []) or [])
            if int(item.get("step", -1)) != int(step)
        ]
        existing = [
            item
            for item in existing_all
            if _eval_checkpoint_step_is_eligible(int(item.get("step", -1)), min_step=min_step)
        ]
        removed_entries = [
            item
            for item in existing_all
            if not _eval_checkpoint_step_is_eligible(int(item.get("step", -1)), min_step=min_step)
        ]
        candidate = _build_eval_checkpoint_entry(step, metrics_summary, metric_name=metric_name)
        if candidate is None:
            print(
                f"[save] skipped checkpoint at step {step}: eval metric {metric_name!r} missing from summary",
                flush=True,
            )
            kept_entries = _topk_eval_checkpoint_entries(existing, top_k=top_k, metric_mode=metric_mode)
        else:
            kept_entries = _topk_eval_checkpoint_entries(existing + [candidate], top_k=top_k, metric_mode=metric_mode)
            should_save = any(int(item.get("step", -1)) == int(step) for item in kept_entries)
            kept_steps = {int(item.get("step", -1)) for item in kept_entries}
            removed_entries.extend(item for item in existing if int(item.get("step", -1)) not in kept_steps)

    should_save = bool(_broadcast_main_process_value(accelerator, should_save))

    if should_save:
        _save_checkpoint(
            accelerator,
            model,
            tokenizer,
            eval_engine,
            project_dir,
            step,
            source_model_path,
            eval_runtime_dtype=_infer_eval_runtime_dtype(eval_engine),
        )
    else:
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        ckpt_dir = project_dir / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for item in removed_entries:
            old_path = ckpt_dir / str(item.get("path") or f"step_{int(item.get('step', -1))}")
            if old_path.exists():
                shutil.rmtree(old_path)
                print(f"[save] pruned checkpoint -> {old_path}", flush=True)
        payload = {
            "metric_name": str(metric_name),
            "metric_mode": str(metric_mode),
            "top_k": int(top_k),
            "min_step": max(1, int(min_step)),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoints": kept_entries,
        }
        _write_json(_checkpoint_registry_path(project_dir), payload)
        if candidate is not None:
            score = float(candidate.get("score", 0.0) or 0.0)
            kept_steps = [int(item.get("step", -1)) for item in kept_entries]
            if should_save:
                print(
                    f"[save] kept eval-top{top_k} checkpoint step={step} {metric_name}={score:.6f} "
                    f"(mode={metric_mode}; kept={kept_steps})",
                    flush=True,
                )
            else:
                print(
                    f"[save] skipped checkpoint at step {step}: {metric_name}={score:.6f} "
                    f"not in eval top{top_k} (kept={kept_steps})",
                    flush=True,
                )
    accelerator.wait_for_everyone()
    return should_save


# --------------------------------------------------------------------------- action training
def _tokenize_prompt(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _normalize_bucket_order(value: Any) -> str:
    if value is None:
        return "shuffle"
    v = str(value).strip().lower()
    if v in {"asc", "ascending", "short_first", "short"}:
        return "asc"
    if v in {"desc", "descending", "long_first", "long"}:
        return "desc"
    if v in {"shuffle", "random"}:
        return "shuffle"
    raise ValueError(f"Invalid bucket_order={value!r}; expected asc, desc, or shuffle")


def _scaled_batch_size_for_ext_len(
    *,
    base_batch_size: int,
    ext_len: int,
    ref_seq_len: int,
    scale_power: float = 2.0,
    anchor_batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    min_batch_size: int = 1,
) -> int:
    base_batch_size = max(1, int(base_batch_size))
    ext_len = max(1, int(ext_len))
    ref_seq_len = max(1, int(ref_seq_len))
    scale_power = max(0.0, float(scale_power))
    anchor_batch_size = max(1, int(base_batch_size if anchor_batch_size is None else anchor_batch_size))
    max_batch_size = max(1, int(base_batch_size if max_batch_size is None else max_batch_size))
    min_batch_size = max(1, int(min_batch_size))
    if max_batch_size < min_batch_size:
        max_batch_size = min_batch_size
    scaled = float(anchor_batch_size) * (float(ref_seq_len) / float(ext_len)) ** scale_power
    return max(min_batch_size, min(max_batch_size, int(round(scaled))))


def _sort_action_batch_groups(
    *,
    built_rows: List[Dict[str, Any]],
    batch_groups: List[List[int]],
    bucket_order: str,
    world_size: int,
    shuffle_seed: Optional[int],
) -> List[List[int]]:
    if not batch_groups:
        return []
    bucket_order = _normalize_bucket_order(bucket_order)
    if bucket_order == "shuffle":
        out = list(batch_groups)
        rng = random.Random(0 if shuffle_seed is None else int(shuffle_seed))
        rng.shuffle(out)
        return out
    reverse = bucket_order == "desc"
    return sorted(
        batch_groups,
        key=lambda ids: (
            int(built_rows[int(ids[0])]["N"]) if ids else 0,
            int(built_rows[int(ids[0])]["L0"]) if ids else 0,
            len(ids),
        ),
        reverse=reverse,
    )


def _pad_and_shard_action_batch_groups(
    *,
    global_batch_groups: List[List[int]],
    world_size: int,
    rank: int,
    built_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    padded = [{"row_indices": list(batch), "is_padding": False} for batch in global_batch_groups]
    pad_count = 0
    if world_size > 1 and padded and len(padded) % world_size != 0:
        pad_count = world_size - (len(padded) % world_size)
        fallback = list(global_batch_groups[0])
        for _ in range(pad_count):
            padded.append({"row_indices": list(fallback), "is_padding": True})

    if not padded:
        return [], 0, 0

    def _active_count(batch: Dict[str, Any]) -> int:
        if bool(batch.get("is_padding", False)):
            return 0
        return sum(
            1
            for row_idx in batch.get("row_indices", [])
            if float(built_rows[int(row_idx)].get("loss_weight", 1.0)) > 0.0
        )

    # NOTE: no per-slot loss rescale is applied for padded ranks. The PPO loss already uses
    # world_size * loss_weight / window_rows (train_action_rows), whose world_size factor
    # cancels FSDP's 1/world_size cross-rank gradient averaging against a GLOBAL row window;
    # padding ranks contribute zero loss_weight (and thus zero gradient), so the surviving
    # ranks' gradients already sum to the correct global mean. A world_size/active_ranks
    # rescale here would double-correct and over-weight padded slots.
    if world_size > 1:
        for slot_start in range(0, len(padded), world_size):
            slot = padded[slot_start : slot_start + world_size]
            active_ranks = max(1, sum(0 if bool(batch["is_padding"]) else 1 for batch in slot))
            slot_active_rows = sum(_active_count(batch) for batch in slot)
            slot_index = slot_start // world_size
            for batch in slot:
                batch["slot_index"] = int(slot_index)
                batch["active_ranks"] = int(active_ranks)
                batch["slot_active_rows"] = int(slot_active_rows)
    else:
        for slot_index, batch in enumerate(padded):
            batch["slot_index"] = int(slot_index)
            batch["active_ranks"] = 1
            batch["slot_active_rows"] = _active_count(batch)

    local = [batch for i, batch in enumerate(padded) if i % max(1, world_size) == rank]
    min_active_ranks = min(int(batch.get("active_ranks", 1)) for batch in padded)
    return local, int(pad_count), int(min_active_ranks)


def _align_fsdp_slots_within_updates(global_groups, *, world_size, built_rows, target_updates):
    """Align workload across ranks without changing which rows update the policy.

    Freeze the original row-balanced optimizer windows first. Sort only within
    each window, preserving its row membership and count of collective slots.
    """
    original, _, _ = _pad_and_shard_action_batch_groups(
        global_batch_groups=global_groups, world_size=world_size, rank=0, built_rows=built_rows)
    plan = _build_row_balanced_update_plan(original, target_updates)
    if plan is None or world_size <= 1:
        return None
    padded = [{"row_indices": list(ids), "is_padding": False} for ids in global_groups]
    while len(padded) < len(original) * world_size:
        padded.append({"row_indices": list(global_groups[0]), "is_padding": True})
    result = []
    begin = 0
    for end, window_rows in zip(plan["boundaries"], plan["window_rows"]):
        window = padded[begin * world_size:end * world_size]
        def cost(entry):
            ids = entry["row_indices"]
            length = int(built_rows[ids[0]]["N"])
            return (len(ids) * length, length, len(ids))
        window.sort(key=cost)
        for offset in range(0, len(window), world_size):
            slot = window[offset:offset + world_size]
            active = sum(not x["is_padding"] for x in slot)
            rows = sum(sum(float(built_rows[i].get("loss_weight", 1.0)) > 0 for i in x["row_indices"])
                       for x in slot if not x["is_padding"])
            for entry in slot:
                entry.update(slot_index=begin + offset // world_size,
                             active_ranks=max(1, active), slot_active_rows=rows,
                             original_update_end=offset + world_size == len(window),
                             original_update_window_rows=window_rows)
            result.extend(slot)
        begin = end
    return result


def _build_row_balanced_update_plan(
    local_batch_groups: List[Dict[str, Any]],
    target_updates_per_outer_step: int,
) -> Optional[Dict[str, Any]]:
    target_updates = max(0, int(target_updates_per_outer_step or 0))
    local_batches = len(local_batch_groups)
    if target_updates <= 0 or local_batches <= 0:
        return None

    slot_rows = [
        max(0, int(batch.get("slot_active_rows", 0) or 0))
        for batch in local_batch_groups
    ]
    total_rows = int(sum(slot_rows))
    if total_rows <= 0:
        return None

    if "original_update_end" in local_batch_groups[0]:
        boundaries = [i + 1 for i, batch in enumerate(local_batch_groups)
                      if batch["original_update_end"]]
        window_rows = [int(local_batch_groups[i - 1]["original_update_window_rows"])
                       for i in boundaries]
        return {"target_updates": target_updates, "effective_updates": len(boundaries),
                "boundaries": boundaries, "boundary_set": set(boundaries),
                "slot_rows": slot_rows, "window_rows": window_rows,
                "step_window_rows": [int(b["original_update_window_rows"]) for b in local_batch_groups],
                "total_rows": total_rows, "target_rows_per_update": total_rows / len(boundaries)}

    positive_slots = sum(1 for row_count in slot_rows if int(row_count) > 0)
    effective_updates = max(1, min(int(target_updates), int(positive_slots), int(local_batches)))
    prefix_rows = [0]
    for row_count in slot_rows:
        prefix_rows.append(prefix_rows[-1] + int(row_count))

    boundaries: List[int] = []
    for update_idx in range(1, effective_updates + 1):
        if update_idx == effective_updates:
            boundary = local_batches
        else:
            target_cumulative = float(total_rows) * float(update_idx) / float(effective_updates)
            min_boundary = (boundaries[-1] + 1) if boundaries else 1
            remaining_updates = effective_updates - update_idx
            max_boundary = local_batches - remaining_updates
            boundary = min_boundary
            while boundary < max_boundary and prefix_rows[boundary] < target_cumulative:
                boundary += 1
            boundary = max(min_boundary, min(boundary, max_boundary))
        boundaries.append(int(boundary))

    cleaned: List[int] = []
    last = 0
    for boundary in boundaries:
        boundary = max(last + 1, min(int(boundary), local_batches))
        cleaned.append(boundary)
        last = boundary
        if boundary >= local_batches:
            break
    if cleaned and cleaned[-1] != local_batches:
        cleaned[-1] = local_batches

    window_rows: List[int] = []
    step_window_rows: List[int] = []
    start = 0
    for boundary in cleaned:
        rows_this_window = int(sum(slot_rows[start:boundary]))
        window_rows.append(rows_this_window)
        step_window_rows.extend([rows_this_window] * max(0, boundary - start))
        start = boundary

    return {
        "target_updates": int(target_updates),
        "effective_updates": int(len(cleaned)),
        "boundaries": cleaned,
        "boundary_set": set(cleaned),
        "slot_rows": slot_rows,
        "window_rows": window_rows,
        "step_window_rows": step_window_rows,
        "total_rows": int(total_rows),
        "target_rows_per_update": float(total_rows) / float(max(len(cleaned), 1)),
    }


def _should_sync_gradients(step_idx_one_based: int, total_steps: int, grad_accum_steps: int) -> bool:
    total_steps = max(0, int(total_steps))
    grad_accum_steps = max(1, int(grad_accum_steps))
    step_idx_one_based = max(1, int(step_idx_one_based))
    return total_steps > 0 and (
        (step_idx_one_based % grad_accum_steps) == 0 or step_idx_one_based == total_steps
    )


def _collate_action_rows(
    built_rows: List[Dict[str, Any]],
    batch_entry: Dict[str, Any],
    *,
    include_old_logp: bool = False,
) -> Dict[str, Any]:
    batch_indices = list(batch_entry.get("row_indices", []))
    if not batch_indices:
        raise RuntimeError("Empty action batch.")
    batch_rows = [built_rows[int(i)] for i in batch_indices]
    if any((r["L0"], r["L1"], r["N"]) != (batch_rows[0]["L0"], batch_rows[0]["L1"], batch_rows[0]["N"]) for r in batch_rows):
        from action_batching import collate_ragged
        entry = dict(batch_entry, pad_id=int(batch_rows[0].get("pad_id", batch_entry.get("pad_id", 0))))
        return collate_ragged(built_rows, entry, include_old_logp)
    L0 = int(batch_rows[0]["L0"])
    L1 = int(batch_rows[0]["L1"])
    N = int(batch_rows[0]["N"])
    for row in batch_rows[1:]:
        if int(row["L0"]) != L0 or int(row["L1"]) != L1 or int(row["N"]) != N:
            raise RuntimeError("Mixed action row shapes inside one microbatch.")

    is_padding = bool(batch_entry.get("is_padding", False))
    rewards = [0.0 if is_padding else float(row["reward"]) for row in batch_rows]
    loss_weights = [0.0 if is_padding else float(row.get("loss_weight", 1.0)) for row in batch_rows]
    out = {
        "ids": torch.tensor(batch_indices, dtype=torch.long),
        "ext": torch.stack([row["ext"] for row in batch_rows]),
        "pmask": torch.stack([row["pmask"] for row in batch_rows]),
        "tok_idx": torch.stack([row["tok_idx"] for row in batch_rows]),
        "labels": torch.stack([row["labels"] for row in batch_rows]),
        "reward": torch.tensor(rewards, dtype=torch.float32),
        "loss_weight": torch.tensor(loss_weights, dtype=torch.float32),
        "loss_denominator": torch.tensor([row.get("loss_denominator", row["L1"]) for row in batch_rows]),
        "L0": L0,
        "L1": L1,
        "N": N,
        "is_padding": is_padding,
        "slot_index": int(batch_entry.get("slot_index", -1)),
        "slot_active_rows": int(batch_entry.get("slot_active_rows", 0) or 0),
        "active_ranks": int(batch_entry.get("active_ranks", 1) or 1),
    }
    if include_old_logp:
        old = []
        for row in batch_rows:
            value = row.get("logp_old")
            if value is None:
                if not is_padding:
                    raise RuntimeError("Missing action logp_old for a non-padding train batch.")
                value = torch.zeros((L0 + L1,), dtype=torch.float32)
            old.append(value)
        out["logp_old"] = torch.stack(old)
    return out


def _build_ref_action_batch_groups(
    *,
    built_rows: List[Dict[str, Any]],
    needed_row_indices: List[int],
    ref_batch_size: int,
    bucket_order: str,
    shuffle_seed: Optional[int],
    token_budget: int = 0,
    attention_budget: int = 0,
) -> List[Dict[str, Any]]:
    # Empty ranks must join the count collective and execute padding forwards.
    # Returning here while another rank has reference rows would deadlock FSDP.
    bucket_rows: Dict[int, List[int]] = {}
    for row_idx in needed_row_indices:
        bucket_rows.setdefault(int(built_rows[int(row_idx)]["bucket_id"]), []).append(int(row_idx))

    groups: List[List[int]] = []
    for bucket_id in sorted(bucket_rows):
        ids = bucket_rows[bucket_id]
        for start in range(0, len(ids), max(1, int(ref_batch_size))):
            groups.append(ids[start : start + max(1, int(ref_batch_size))])
    groups = _sort_action_batch_groups(
        built_rows=built_rows,
        batch_groups=groups,
        bucket_order=bucket_order,
        world_size=1,
        shuffle_seed=shuffle_seed,
    )
    if token_budget > 0:
        from action_batching import pack_ids
        groups = pack_ids(needed_row_indices, built_rows, token_budget=token_budget, max_rows=ref_batch_size, attention_budget=attention_budget)
    out = [{"row_indices": list(batch), "is_padding": False} for batch in groups]
    if dist.is_available() and dist.is_initialized():
        gathered_counts = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_counts, int(len(out)))
        target = max(int(v or 0) for v in gathered_counts)
        fallback = list(groups[0]) if groups else [0]
        while len(out) < target:
            out.append({"row_indices": list(fallback), "is_padding": True})
    return out


def _build_bucketed_action_rows(
    *,
    rows: List[Dict[str, Any]],
    tokenizer,
    config,
    mask_id: int,
    pad_id: int,
    device,
    world_size: int,
    rank: int,
    block_size: int,
    shrink: int,
    max_prompt_len: int,
    rollout_bucket_size: int,
    bucket_order: str,
    shuffle_seed: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, Dict[str, Any]]:
    raw_records: List[Dict[str, Any]] = []
    dropped_prompt = 0
    dropped_bad = 0
    build_t0 = time.time()

    for row_idx, row in enumerate(rows):
        prompt_ids = (list(row["prompt_token_ids"]) if "prompt_token_ids" in row
                      else _tokenize_prompt(tokenizer, row["prompt_text"]))
        if bool(config.training.get("require_complete_action_trace", False)) and row.get("trace_version") != 2:
            raise ValueError("Training requires complete action traces")
        if row.get("trace_version") == 2:
            context_ids = row["conditioning_token_ids"]
            context_steps = row["conditioning_step_map"]
            target_mask = row["action_loss_mask"]
            if not (len(context_ids) == len(context_steps) == len(target_mask)):
                raise ValueError("Incomplete or misaligned terminal-block replay")
            if (len(prompt_ids) + len(context_ids)) % block_size:
                raise ValueError("Terminal-block replay must end on an absolute block boundary")
            if any(int(t) != mask_id and int(step) < 0 for t, step in zip(context_ids, context_steps)):
                raise ValueError("A resolved generated token has no recorded diffusion step")
            expected_t = float(config.get("rollout", {}).get("temperature", 0.8))
            if float(row["sampling_temperature"]) != expected_t:
                raise ValueError("Recorded rollout temperature differs from the learner policy")
            if set(row["banned_token_ids"]) != {mask_id, L.EXPAND_TOKEN_ID}:
                raise ValueError("Recorded action vocabulary support differs from the learner")
        if len(prompt_ids) > max_prompt_len:
            dropped_prompt += 1
            continue
        resp_ids = list(row.get("response_token_ids") or [])
        step_map = list(row.get("step_map") or [])
        if (not resp_ids and not row.get("conditioning_token_ids")) or len(resp_ids) != len(step_map):
            dropped_bad += 1
            continue
        raw_records.append(
            {
                "row_idx": int(row_idx),
                "prompt_ids": prompt_ids,
                "resp_ids": resp_ids,
                "step_map": step_map,
                "conditioning_ids": row.get("conditioning_token_ids"),
                "conditioning_steps": row.get("conditioning_step_map"),
                "action_loss_mask": row.get("action_loss_mask"),
                "reward": float(row.get("reward", 0.0)),
                "loss_weight": float(row.get("loss_weight", 1.0)),
                "prompt_len": len(prompt_ids),
                "resp_len": len(resp_ids),
                "sort_len": len(prompt_ids) + 2 * len(resp_ids),
            }
        )

    if not raw_records:
        return [], [], 0, {
            "raw_rows": 0,
            "dropped_prompts": int(dropped_prompt),
            "dropped_bad": int(dropped_bad),
            "build_elapsed_sec": float(time.time() - build_t0),
        }

    ordered = sorted(raw_records, key=lambda r: (int(r["sort_len"]), int(r["prompt_len"]), int(r["resp_len"])))
    rollout_bucket_size = max(1, int(rollout_bucket_size))
    bucket_order = _normalize_bucket_order(bucket_order)
    train_batch_anchor_seq_len = max(1, int(config.training.get("train_batch_anchor_seq_len", 4000)))
    train_batch_anchor_size = max(1, int(config.training.get("train_batch_anchor_size", config.training.batch_size_lm)))
    train_batch_size_cap = max(1, int(config.training.get("train_batch_size_max", config.training.batch_size_lm)))
    train_batch_size_scale_power = max(0.0, float(config.training.get("train_batch_size_scale_power", 2.0)))
    base_batch_size = max(1, int(config.training.batch_size_lm))

    built_rows: List[Dict[str, Any]] = []
    global_train_groups: List[List[int]] = []
    bucket_ext_lens: List[int] = []
    clean_lens: List[int] = []
    prompt_lens: List[int] = []
    resp_lens: List[int] = []
    train_batch_sizes: List[int] = []
    bucket_count = 0

    for bucket_id, start in enumerate(range(0, len(ordered), rollout_bucket_size)):
        bucket = ordered[start : start + rollout_bucket_size]
        if not bucket:
            continue
        # Bucket lengths set the loss denominator; row token positions stay fixed.
        bucket_response_length = max(int(r["resp_len"]) for r in bucket)
        row_start = len(built_rows)
        for rec in bucket:
            out, L0, L1 = L.collect_action_segment(
                rec["prompt_ids"], rec["resp_ids"], rec["step_map"],
                reward=rec["reward"], loss_weight=rec["loss_weight"],
                block_size=block_size, mask_id=mask_id, shrink=shrink,
                method=str(config.training.method),
                conditioning_ids=rec["conditioning_ids"],
                conditioning_steps=rec["conditioning_steps"],
                action_loss_mask=rec["action_loss_mask"],
            )
            if out is None:
                continue
            ext, pmask, tok_idx, labels, rewards_out, lws_out = out
            ext_len = int(ext.shape[1])
            for i in range(ext.shape[0]):
                built_rows.append(
                    {
                        "ext": ext[i], "pmask": pmask[i], "tok_idx": tok_idx[i],
                        "labels": labels[i], "reward": float(rewards_out[i]),
                        "loss_weight": float(lws_out[i]),
                        "L0": int(L0), "L1": int(L1), "N": ext_len,
                        "prompt_length": int(rec["prompt_len"]),
                        "response_length": int(rec["resp_len"]),
                        "loss_denominator": bucket_response_length,
                        "source_row_idx": int(rec["row_idx"]), "pad_id": int(pad_id),
                        "bucket_id": int(bucket_id),
                    }
                )
        bucket_row_ids = list(range(row_start, len(built_rows)))
        if not bucket_row_ids:
            continue
        max_ext_len = max(built_rows[i]["N"] for i in bucket_row_ids)
        effective_bs = _scaled_batch_size_for_ext_len(
            base_batch_size=base_batch_size, ext_len=max_ext_len,
            ref_seq_len=train_batch_anchor_seq_len, scale_power=train_batch_size_scale_power,
            anchor_batch_size=train_batch_anchor_size, max_batch_size=train_batch_size_cap,
        )
        for i in bucket_row_ids:
            built_rows[i]["base_batch_size_lm"] = int(effective_bs)
        train_batch_sizes.append(int(effective_bs))
        bucket_ext_lens.append(max_ext_len)
        clean_lens.append(max(built_rows[i]["L0"] + built_rows[i]["L1"] for i in bucket_row_ids))
        prompt_lens.append(max(rec["prompt_len"] for rec in bucket))
        resp_lens.append(bucket_response_length)
        for group_start in range(0, len(bucket_row_ids), int(effective_bs)):
            global_train_groups.append(bucket_row_ids[group_start : group_start + int(effective_bs)])
        bucket_count += 1

    if not built_rows or not global_train_groups:
        return [], [], 0, {
            "raw_rows": int(len(raw_records)),
            "dropped_prompts": int(dropped_prompt),
            "dropped_bad": int(dropped_bad),
            "build_elapsed_sec": float(time.time() - build_t0),
        }

    global_train_groups = _sort_action_batch_groups(
        built_rows=built_rows,
        batch_groups=global_train_groups,
        bucket_order=bucket_order,
        world_size=world_size,
        shuffle_seed=shuffle_seed,
    )
    local_train_groups, pad_count, min_active_ranks = _pad_and_shard_action_batch_groups(
        global_batch_groups=global_train_groups,
        world_size=world_size,
        rank=rank,
        built_rows=built_rows,
    )
    token_budget = int(config.training.get("train_token_budget", 0))
    if token_budget > 0:
        from action_batching import pack_training_windows
        plan = _build_row_balanced_update_plan(local_train_groups,
            int(config.training.get("target_updates_per_outer_step", 0)))
        if plan is None:
            raise ValueError("Token packing requires row-balanced optimizer windows")
        packed = pack_training_windows(global_train_groups, built_rows, plan,
            world_size=world_size, pad_id=pad_id, token_budget=token_budget,
            max_rows=int(config.training.get("packed_batch_size_max", 32)),
            attention_budget=int(config.training.get("attention_element_budget", 0)))
        local_train_groups = packed[rank::world_size]
        pad_count = sum(entry["is_padding"] for entry in packed)
        min_active_ranks = min(entry["active_ranks"] for entry in packed)
        train_batch_sizes = [len(entry["row_indices"]) for entry in packed if not entry["is_padding"]]
    elif bool(config.training.get("align_fsdp_workloads", True)):
        aligned = _align_fsdp_slots_within_updates(global_train_groups,
            world_size=world_size, built_rows=built_rows,
            target_updates=int(config.training.get("target_updates_per_outer_step", 0)))
        if aligned is not None:
            local_train_groups = aligned[rank::world_size]

    stats = {
        "raw_rows": int(len(raw_records)),
        "built_2n_rows": int(len(built_rows)),
        "bucket_count": int(bucket_count),
        "dropped_prompts": int(dropped_prompt),
        "dropped_bad": int(dropped_bad),
        "global_train_batches": int(len(local_train_groups) * world_size),
        "sync_padding_batches": int(pad_count),
        "min_active_ranks_per_slot": int(min_active_ranks),
        "bucket_order": str(bucket_order),
        "bucket_shuffle_seed": None if shuffle_seed is None else int(shuffle_seed),
        "rollout_bucket_size": int(rollout_bucket_size),
        "train_batch_anchor_seq_len": int(train_batch_anchor_seq_len),
        "train_batch_anchor_size": int(train_batch_anchor_size),
        "train_batch_size_cap": int(train_batch_size_cap),
        "train_batch_size_scale_power": float(train_batch_size_scale_power),
        "train_batch_size_min": int(min(train_batch_sizes)) if train_batch_sizes else base_batch_size,
        "train_batch_size_max": int(max(train_batch_sizes)) if train_batch_sizes else base_batch_size,
        "seq_len_total_mean": float(sum(bucket_ext_lens) / len(bucket_ext_lens)) if bucket_ext_lens else 0.0,
        "seq_len_total_max": int(max(bucket_ext_lens)) if bucket_ext_lens else 0,
        "clean_len_mean": float(sum(clean_lens) / len(clean_lens)) if clean_lens else 0.0,
        "clean_len_max": int(max(clean_lens)) if clean_lens else 0,
        "prompt_len_mean": float(sum(prompt_lens) / len(prompt_lens)) if prompt_lens else 0.0,
        "response_len_mean": float(sum(resp_lens) / len(resp_lens)) if resp_lens else 0.0,
        "min_batch_ext_len": int(min(bucket_ext_lens)) if bucket_ext_lens else 0,
        "max_batch_ext_len": int(max(bucket_ext_lens)) if bucket_ext_lens else 0,
        "build_elapsed_sec": float(time.time() - build_t0),
    }
    return built_rows, local_train_groups, int(len(local_train_groups) * world_size), stats


def train_action_rows(
    accelerator: Accelerator,
    *,
    model,
    optimizer,
    rows: List[Dict[str, Any]],
    tokenizer,
    config,
    mask_id: int,
    pad_id: int,
    world_size: int,
    rank: int,
    sft_anchor,
) -> Dict[str, float]:
    """Bucket action rows, cache logp_old in microbatches, then run one PPO pass.

    `target_updates_per_outer_step` controls optimizer sync/update boundaries; it is
    not an epoch count. Batch size knobs are applied when forming the microbatches.
    """
    block_size = int(config.training.block_size)
    shrink = int(config.training.shrink)
    from action_policy import validate_temperature
    temperature = validate_temperature(config.get("rollout", {}).get("temperature", 0.8))
    banned_token_ids = (int(mask_id), L.EXPAND_TOKEN_ID)
    eps = float(config.training.eps)
    beta = float(config.training.beta)
    use_k3 = bool(config.training.use_kl_estimator_k3)
    max_prompt_len = int(config.training.get("max_prompt_len", 8192))
    target_updates_per_outer_step = max(0, int(config.training.get("target_updates_per_outer_step", 0) or 0))
    grad_accum = max(1, int(config.training.gradient_accumulation_steps))
    ref_batch_size_lm = max(1, int(config.training.get("ref_batch_size_lm", 1)))
    rollout_bucket_size = max(1, int(config.training.get("rollout_bucket_size", 8)))
    bucket_order = _normalize_bucket_order(
        config.training.get("bucket_order", os.environ.get("TRACERL_BUCKET_ORDER", "shuffle"))
    )
    base_seed = int(config.training.get("seed", 0) or 0)
    bucket_shuffle_seed = base_seed + int(config.get("seed_offset", 0) or 0)
    device = accelerator.device
    dtype = getattr(model, "_rl_attention_dtype", getattr(model, "dtype", torch.bfloat16))

    metrics = {
        "action_policy_loss": 0.0,
        "action_kl_loss": 0.0,
        "action_rows": 0.0,
        "action_updates": 0.0,
    }
    _progress_print(
        accelerator,
        rank,
        "action-build",
        (
            f"start global_rows={len(rows)} batch_size_lm={int(config.training.batch_size_lm)} "
            f"ref_batch_size_lm={ref_batch_size_lm} bucket_size={rollout_bucket_size} "
            f"bucket_order={bucket_order}"
        ),
        env_name="RL_TRAIN_PROGRESS_RANKS",
        default_ranks="main",
    )

    build_t0 = time.time()
    built, local_train_batches, global_train_batches, bucket_stats = _build_bucketed_action_rows(
        rows=rows,
        tokenizer=tokenizer,
        config=config,
        mask_id=mask_id,
        pad_id=pad_id,
        device=device,
        world_size=world_size,
        rank=rank,
        block_size=block_size,
        shrink=shrink,
        max_prompt_len=max_prompt_len,
        rollout_bucket_size=rollout_bucket_size,
        bucket_order=bucket_order,
        shuffle_seed=bucket_shuffle_seed,
    )
    local_real_row_ids = sorted(
        {
            int(row_idx)
            for batch in local_train_batches
            if not bool(batch.get("is_padding", False))
            for row_idx in batch.get("row_indices", [])
        }
    )
    metrics["action_rows"] = float(len(local_real_row_ids))
    _progress_print(
        accelerator,
        rank,
        "action-build",
        (
            f"done raw={int(bucket_stats.get('raw_rows', 0))}/{len(rows)} "
            f"built_2n={len(built)} local_train_batches={len(local_train_batches)} "
            f"global_train_batches={global_train_batches} local_2n_rows={len(local_real_row_ids)} "
            f"train_bs={int(bucket_stats.get('train_batch_size_min', 0))}:"
            f"{int(bucket_stats.get('train_batch_size_max', 0))} "
            f"pad_batches={int(bucket_stats.get('sync_padding_batches', 0))} "
            f"elapsed={_fmt_seconds(time.time() - build_t0)}"
        ),
        env_name="RL_TRAIN_PROGRESS_RANKS",
        default_ranks="all",
    )
    if _is_main(accelerator):
        print(
            "[Action][Bucketed] "
            f"order={bucket_stats.get('bucket_order', bucket_order)} "
            f"shuffle_seed={bucket_stats.get('bucket_shuffle_seed')} "
            f"bucket_count={int(bucket_stats.get('bucket_count', 0))} "
            f"rollout_bucket_size={int(bucket_stats.get('rollout_bucket_size', rollout_bucket_size))} "
            f"dropped_prompt={int(bucket_stats.get('dropped_prompts', 0))} "
            f"dropped_bad={int(bucket_stats.get('dropped_bad', 0))} "
            f"seq_len_mean={float(bucket_stats.get('seq_len_total_mean', 0.0)):.1f} "
            f"seq_len_max={int(bucket_stats.get('seq_len_total_max', 0))} "
            f"clean_len_mean={float(bucket_stats.get('clean_len_mean', 0.0)):.1f} "
            f"clean_len_max={int(bucket_stats.get('clean_len_max', 0))} "
            f"train_bs_base={int(config.training.batch_size_lm)} "
            f"train_bs_anchor={int(bucket_stats.get('train_batch_anchor_size', 0))} "
            f"train_bs_cap={int(bucket_stats.get('train_batch_size_cap', 0))} "
            f"train_bs_min={int(bucket_stats.get('train_batch_size_min', 0))} "
            f"train_bs_max={int(bucket_stats.get('train_batch_size_max', 0))} "
            f"ref_bs={ref_batch_size_lm} "
            f"sync_padding_batches={int(bucket_stats.get('sync_padding_batches', 0))} "
            f"min_active_ranks={int(bucket_stats.get('min_active_ranks_per_slot', 0))}",
            flush=True,
        )

    if not built or not local_train_batches:
        model.train()
        optimizer.zero_grad()
        return sft_anchor.finish_updates(model, optimizer, metrics)

    reuse_first_window = bool(config.training.get("reuse_initial_window_logp", False))
    initial_plan = _build_row_balanced_update_plan(local_train_batches, target_updates_per_outer_step)
    first_window_end = initial_plan["boundaries"][0] if reuse_first_window and initial_plan else 0
    if reuse_first_window:
        if not initial_plan or not bool(config.training.get("fused_moe_parameters", False)):
            raise ValueError("Initial reference reuse requires fused deterministic MoE and fixed update windows")
        core_config = accelerator.unwrap_model(model).config
        if any(float(getattr(core_config, key, 0.) or 0.) != 0. for key in
               ("attention_dropout", "embedding_dropout", "output_dropout")):
            raise ValueError("Initial reference reuse is invalid when train/eval dropout differs")
        if any(isinstance(module, torch.nn.Dropout) and module.p > 0 for module in model.modules()):
            raise ValueError("Initial reference reuse requires zero dropout")
        _log(accelerator, f"[Action][Reference reuse] first optimizer window: {initial_plan['window_rows'][0]} rows; remaining windows retain frozen start-of-step logp")
    needed_row_indices = sorted(
        {
            int(row_idx)
            for batch in local_train_batches[first_window_end:]
            for row_idx in batch.get("row_indices", [])
        }
    )
    ref_batch_source = _build_ref_action_batch_groups(
        built_rows=built,
        needed_row_indices=needed_row_indices,
        ref_batch_size=ref_batch_size_lm,
        bucket_order=bucket_order,
        shuffle_seed=bucket_shuffle_seed,
        token_budget=int(config.training.get("ref_token_budget", 0)),
        attention_budget=int(config.training.get("attention_element_budget", 0)),
    )
    ref_local_batches = len(ref_batch_source)
    row_balanced_update_plan = _build_row_balanced_update_plan(
        local_batch_groups=local_train_batches,
        target_updates_per_outer_step=target_updates_per_outer_step,
    )
    if _is_main(accelerator):
        if row_balanced_update_plan is not None:
            print(
                "[Action][DynamicAccum] "
                f"target_updates={row_balanced_update_plan['target_updates']} "
                f"effective_updates={row_balanced_update_plan['effective_updates']} "
                f"total_rows={row_balanced_update_plan['total_rows']} "
                f"target_rows_per_update={row_balanced_update_plan['target_rows_per_update']:.2f} "
                f"boundaries={row_balanced_update_plan['boundaries']} "
                f"window_rows={row_balanced_update_plan['window_rows']}",
                flush=True,
            )
        elif target_updates_per_outer_step > 0:
            print(
                "[Action][DynamicAccum] requested but unavailable; falling back to gradient_accumulation_steps.",
                flush=True,
            )

    # --- cache logp_old once (frozen reference for this outer step) ---
    model.eval()
    cache_t0 = time.time()
    _progress_print(
        accelerator,
        rank,
        "action-ref-logp",
        (
            f"start batches={ref_local_batches} real_train_batches={len(local_train_batches)} "
            f"ref_batch_size_lm={ref_batch_size_lm} local_2n_rows={int(metrics['action_rows'])}"
        ),
        env_name="RL_TRAIN_PROGRESS_RANKS",
        default_ranks="all",
    )
    attention_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _attention_for(N: int, L0: int) -> torch.Tensor:
        if torch.is_tensor(L0):
            return None
        key = (int(N), int(L0))
        att = attention_cache.get(key)
        if att is None:
            att = L.make_basic_block_attention(int(N), int(L0), block_size, dtype=dtype).to(device)
            attention_cache[key] = att
        return att

    ref_bar = _progress_bar(
        ref_batch_source,
        accelerator,
        rank,
        desc=f"Action RefLogP r{rank}",
        total=ref_local_batches,
        leave=False,
        default_ranks="all",
    )
    from performance import MicrobatchProfiler
    ref_profiler = MicrobatchProfiler("reference", rank)
    for cache_idx, batch_entry in enumerate(ref_bar, start=1):
        is_padding_ref = bool(batch_entry.get("is_padding", False))
        batch = _collate_action_rows(built, batch_entry, include_old_logp=False)
        ext = batch["ext"].to(device, non_blocking=True)
        tok_idx = batch["tok_idx"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        att = _attention_for(batch["N"], batch["L0"])
        logp_old = L.compute_action_logp_old_tok(
            model=model,
            extended_input_ids=ext,
            tok_idx_ext=tok_idx,
            labels=labels,
            p_mask=batch["pmask"].to(device, non_blocking=True),
            block_size=block_size,
            temperature=temperature, banned_token_ids=banned_token_ids,
            basic_block_attention=att,
            pad_id=pad_id,
            start_pos=batch["L0"],
            L0=batch["L0"],
            L1=batch["L1"],
        )
        if not is_padding_ref:
            # One device-to-host transfer per batch, instead of one per row.
            logp_old_cpu = logp_old.detach().float().cpu()
            for local_idx, row_idx in enumerate(batch["ids"].tolist()):
                length = built[int(row_idx)]["L0"] + built[int(row_idx)]["L1"]
                built[int(row_idx)]["logp_old"] = logp_old_cpu[local_idx, :length]
        ref_profiler.step()
        ref_bar.set_postfix({"batch": f"{cache_idx}/{ref_local_batches}", "B": int(ext.shape[0])}, refresh=False)
    ref_bar.close()
    ref_profiler.close()
    _progress_print(
        accelerator,
        rank,
        "action-ref-logp",
        f"done batches={ref_local_batches} elapsed={_fmt_seconds(time.time() - cache_t0)}",
        env_name="RL_TRAIN_PROGRESS_RANKS",
        default_ranks="all",
    )

    accelerator.wait_for_everyone()

    # --- one PPO pass over action microbatches ---
    model.train()
    n_losses = 0
    metric_sums = torch.zeros(2, device=device, dtype=torch.float64)
    total_train_forwards = len(local_train_batches)
    train_t0 = time.time()
    optimizer.zero_grad()
    _progress_print(
        accelerator,
        rank,
        "action-ppo",
        (
            f"start batches={total_train_forwards} grad_accum={grad_accum} "
            f"target_updates={target_updates_per_outer_step or 'off'}"
        ),
        env_name="RL_TRAIN_PROGRESS_RANKS",
        default_ranks="all",
    )
    train_bar = _progress_bar(
        local_train_batches,
        accelerator,
        rank,
        desc=f"Action PPO r{rank}",
        total=total_train_forwards,
        leave=False,
        default_ranks="all",
    )
    train_profiler = MicrobatchProfiler("ppo", rank)
    for idx, batch_entry in enumerate(train_bar):
        step_idx = idx + 1
        if row_balanced_update_plan is not None:
            should_sync = step_idx in row_balanced_update_plan["boundary_set"]
        else:
            should_sync = _should_sync_gradients(step_idx, total_train_forwards, grad_accum)
        shard_each_microbatch = bool(config.training.get("shard_accumulated_gradients", False))
        sync_context = (contextlib.nullcontext() if should_sync or shard_each_microbatch
                        else accelerator.no_sync(model))
        reuse_current_reference = idx < first_window_end
        batch = _collate_action_rows(built, batch_entry, include_old_logp=not reuse_current_reference)
        is_padding_batch = bool(batch.get("is_padding", False))
        active_samples = 0.0 if is_padding_batch else float(batch["loss_weight"].sum())
        ext = batch["ext"].to(device, non_blocking=True)
        pmask = batch["pmask"].to(device, non_blocking=True)
        tok_idx = batch["tok_idx"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        adv = batch["reward"].to(device, non_blocking=True)
        loss_weight = batch["loss_weight"].to(device, non_blocking=True)
        old_lp = None if reuse_current_reference else batch["logp_old"].to(device, non_blocking=True)
        att = _attention_for(batch["N"], batch["L0"])
        with sync_context:
            total, pl, kl = L.action_ppo_loss(
                model=model,
                extended_input_ids=ext,
                p_mask=pmask,
                tok_idx_ext=tok_idx,
                labels=labels,
                adv=adv,
                loss_weight=loss_weight,
                logp_old_tok=old_lp,
                basic_block_attention=att,
                pad_id=pad_id,
                start_pos=batch["L0"],
                L0=batch["L0"],
                L1=batch["L1"],
                eps=eps, beta=beta, use_kl_estimator_k3=use_k3,
                block_size=block_size,
                reuse_current_as_old=reuse_current_reference,
                temperature=temperature, banned_token_ids=banned_token_ids,
                loss_denominator=batch["loss_denominator"].to(device, non_blocking=True),
            )
            if row_balanced_update_plan is not None:
                window_rows = max(1, int(row_balanced_update_plan["step_window_rows"][idx]))
                dynamic_weight = float(world_size) * active_samples / float(window_rows)
                effective_loss = total * dynamic_weight
            else:
                effective_loss = total / float(grad_accum)
            accelerator.backward(effective_loss)

        train_profiler.step()
        if active_samples > 0:
            metric_sums.add_(torch.stack((pl.detach(), kl.detach())).to(torch.float64))
            n_losses += 1
        if should_sync:
            sft_anchor.accumulate_update(model, int(metrics["action_updates"]))
            if config.training.max_grad_norm:
                accelerator.clip_grad_norm_(model.parameters(), float(config.training.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad()
            metrics["action_updates"] += 1.0
        train_bar.set_postfix(
            {
                "upd": int(metrics["action_updates"]),
                "B": int(ext.shape[0]),
            },
            refresh=False,
        )
    train_bar.close()
    train_profiler.close()
    _progress_print(
        accelerator,
        rank,
        "action-ppo",
        f"done batches={total_train_forwards} updates={int(metrics['action_updates'])} elapsed={_fmt_seconds(time.time() - train_t0)}",
        env_name="RL_TRAIN_PROGRESS_RANKS",
        default_ranks="all",
    )
    if n_losses:
        mean_metrics = (metric_sums / n_losses).cpu().tolist()
        metrics["action_policy_loss"], metrics["action_kl_loss"] = mean_metrics
    return sft_anchor.finish_updates(model, optimizer, metrics)


def _wandb_log(accelerator: Accelerator, metrics: Dict[str, Any], outer_step: int) -> None:
    """Log a metrics dict to the active wandb run. No-op off main / if no run is active."""
    if not accelerator.is_main_process:
        return
    try:
        import wandb
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return
    payload = dict(metrics)
    payload["outer_step"] = int(outer_step)
    try:
        wandb.log(payload, step=int(outer_step))
    except Exception as exc:  # noqa: BLE001 - logging must never kill training
        print(f"[wandb] log failed at step {outer_step}: {exc}", flush=True)


def _add_eval_wandb_metrics(payload: Dict[str, Any], eval_metrics: Dict[str, Any]) -> None:
    _ov = eval_metrics.get("overall", {}) or {}
    payload["eval/hotpot_2wiki_avg_f1"] = float(_ov.get("hotpot_2wiki_avg_f1", 0.0) or 0.0)
    payload["eval/overall_mean_f1"] = float(_ov.get("mean_f1", 0.0) or 0.0)
    payload["eval/overall_mean_em"] = float(_ov.get("mean_em", 0.0) or 0.0)
    payload["eval/overall_mean_subem"] = float(_ov.get("mean_subem", 0.0) or 0.0)

    names_by_canonical = {
        "hotpotqa": "HotpotQA_val",
        "2wiki": "2wiki_val",
    }
    for ds_name, ds_metrics in eval_metrics.items():
        if not isinstance(ds_metrics, dict):
            continue
        canonical = _dataset_key(ds_name)
        wandb_name = names_by_canonical.get(canonical)
        if not wandb_name:
            continue
        payload[f"eval/{wandb_name}_mean_f1"] = float(ds_metrics.get("mean_f1", 0.0) or 0.0)
        payload[f"eval/{wandb_name}_mean_em"] = float(ds_metrics.get("mean_em", 0.0) or 0.0)
        payload[f"eval/{wandb_name}_mean_subem"] = float(ds_metrics.get("mean_subem", 0.0) or 0.0)


# --------------------------------------------------------------------------- main
def main():
    config = get_config()
    project_name = config.experiment.project

    # wandb: offline by default (writes run files to ./wandb/); override with
    # WANDB_MODE=online to stream, or WANDB_MODE=disabled to turn off entirely.
    os.environ.setdefault("WANDB_MODE", "offline")
    wandb_enabled = os.environ.get("WANDB_MODE", "offline").strip().lower() != "disabled"

    accelerator = Accelerator(
        gradient_accumulation_steps=1,  # we drive accumulation manually per stream
        mixed_precision=config.training.mixed_precision,
        split_batches=True,
    )
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if config.training.seed is not None:
        set_seed(int(config.training.seed))

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    project_dir = _prepare_project_dir(
        accelerator,
        project_name,
        start_from_scratch=bool(config.experiment.get("start_from_scratch", False)),
    )
    from performance import install_wall_timers, write_run_manifest
    run_started_at = install_wall_timers(globals(), project_dir, rank)
    if accelerator.is_main_process:
        write_run_manifest(project_dir, config, run_started_at)
    pretrained_model = config.model.pretrained_model
    # ---- init wandb directly (main process only) ----
    # We call wandb.init ourselves instead of accelerator.init_trackers: in offline mode
    # accelerate's WandBTracker.store_init_configuration finishes the run and re-inits it to
    # bake in the config (tracking.py), leaving a stray empty "shell" run dir. A single
    # wandb.init(config=...) produces exactly one run.
    if wandb_enabled and _is_main(accelerator):
        try:
            import wandb
            from omegaconf import OmegaConf

            try:
                tracker_config = OmegaConf.to_container(config, resolve=True)
            except Exception:
                tracker_config = {}
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "diffuplan-rl"),
                name=os.environ.get("WANDB_NAME", f"{project_name}_{int(time.time())}"),
                config=tracker_config,
                mode=os.environ.get("WANDB_MODE", "offline"),
                dir=str(project_dir),
            )
            _log(
                accelerator,
                f"[Setup] wandb initialized (mode={os.environ.get('WANDB_MODE')}, "
                f"project={os.environ.get('WANDB_PROJECT', 'diffuplan-rl')}, dir={project_dir})",
            )
        except Exception as exc:  # noqa: BLE001 - wandb must never block training
            _log(accelerator, f"[Setup] wandb init skipped ({type(exc).__name__}: {exc})")
    if _is_main(accelerator) and world_size > 1:
        _log(accelerator, f"[Setup] non-rank0 stdout/stderr -> {_RL_DIR / 'logs' / 'rank_i.out'}")

    _log(accelerator, f"[Setup] loading learner model from {pretrained_model}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    # The checkpoint's config.json predates the use_qk_norm flag, but the SFT weights DO
    # contain query/key layernorm tensors -> the model must be built with use_qk_norm=True
    # or transformers silently drops those trained weights. Detect from the weight index.
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)
    if not hasattr(model_config, "use_qk_norm"):
        has_qk = _checkpoint_has_qk_norm(pretrained_model)
        setattr(model_config, "use_qk_norm", bool(has_qk))
        _log(accelerator, f"[Setup] backfilled config.use_qk_norm={bool(has_qk)} (from checkpoint weights)")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model, config=model_config, trust_remote_code=True,
        torch_dtype="auto", attn_implementation="sdpa",
    )
    if str(config.training.mixed_precision) == "bf16":
        model._rl_attention_dtype = torch.bfloat16
    if hasattr(model, "config"):
        model.config.fuse_cross_entropy = True
        model.config.use_cache = False
    from learner_optimizations import install_learner_optimizations
    learner_optimizations = install_learner_optimizations(
        model,
        sparse_action_logits=bool(config.training.get("sparse_action_logits", True)),
        grouped_reference_moe=bool(config.training.get("grouped_reference_moe", True)),
        logp_chunk_size=int(config.training.get("logp_chunk_size", 128)),
    )
    _log(accelerator, f"[Setup] learner optimizations: {learner_optimizations}")
    if bool(config.training.get("fused_moe_parameters", False)):
        from fused_experts import fuse_expert_parameters
        _log(accelerator, f"[Setup] fused learner expert parameters: {fuse_expert_parameters(model)}")
    if bool(config.training.get("sm100_expert_gemm", False)):
        if not bool(config.training.get("fused_moe_parameters", False)):
            raise ValueError("sm100_expert_gemm requires fused_moe_parameters")
        from sm100_expert_gemm import install_sm100_expert_gemm
        install_sm100_expert_gemm(model, verify=bool(config.training.get("verify_expert_gemm", False)))
        _log(accelerator, "[Setup] native SM100 BF16 expert forward and backward GEMMs")
    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()

    mask_id = int(getattr(model.config, "mask_token_id", 156895) or 156895)
    pad_id = int(getattr(model.config, "pad_token_id", 156892) or 156892)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.optimizer.params.learning_rate))
    learner_model_config = model.config
    model, optimizer = accelerator.prepare(model, optimizer)

    from sft_anchor import FrozenSFTAnchor
    sft_anchor = FrozenSFTAnchor(accelerator, config, trainer=sys.modules[__name__],
                                optimizer=optimizer)

    # Build the shared dInfer rollout engine (action + edit) on every rank.
    from diffuplan_core.dinfer_runtime.dinfer_rollout_engine import DInferRolloutEngine, DInferRuntimeConfig

    dinfer_cfg = config.get("dinfer", {})
    eval_cfg = config.get("evaluation", {})
    eval_before_step_every = int(eval_cfg.get("eval_before_step_every", 0) or 0)
    runtime_init_threshold = float(config.rollout.get("confidence_threshold", 0.9))
    if eval_before_step_every > 0:
        runtime_init_threshold = float(eval_cfg.get("confidence_threshold", runtime_init_threshold))
    runtime_cfg = DInferRuntimeConfig(
        use_compile=bool(dinfer_cfg.get("use_compile", True)),
        deterministic_zero_temp=bool(dinfer_cfg.get("deterministic_zero_temp", True)),
        prefilling_limit=int(dinfer_cfg.get("prefilling_limit", 128)),
        dist_timeout_sec=int(dinfer_cfg.get("dist_timeout_sec", 1800)),
    )
    dinfer_engine = DInferRolloutEngine(
        model_path=pretrained_model,
        retriever_url=config.search.get("url", "http://127.0.0.1:8000/retrieve"),
        topk_search=int(config.search.get("topk", 3)),
        max_batch_size=int(config.rollout.get("rollout_gpu_batch_size", 1)),
        length=int(config.rollout.get("max_token", 2048)),
        block_length=int(config.rollout.get("block_size", 32)),
        threshold=runtime_init_threshold,
        max_turns=1,  # editable-plan rollout owns its own multi-turn loop
        max_obs_length=int(config.search.get("max_obs_length", 500)),
        runtime_cfg=runtime_cfg,
        # Each DP rank generates its task shard with a complete model replica.
        parallel_mode="dp",
        unload_after_rollout=False,
        expert_map_path=str(_DIFFUPLAN_ROOT),
        tokenizer=tokenizer,
        model_config=learner_model_config,
    )
    dinfer_engine.editable_batch_size = int(config.rollout.get("editable_batch_size", 1))
    dinfer_engine.interleave_tasks = bool(config.rollout.get("interleave_tasks", False))
    dinfer_engine.rollout_seed = int(config.training.get("seed", 10086))
    dinfer_engine.batch_edit_sampling = bool(config.rollout.get("batch_edit_sampling", False))
    dinfer_engine.b200_moe_tiles = bool(dinfer_cfg.get("b200_moe_tiles", False))
    dinfer_engine.optimize_decode = bool(dinfer_cfg.get("optimize_decode", False))
    dinfer_engine.full_prompt_prefill = bool(dinfer_cfg.get("full_prompt_prefill", False))
    t0 = time.time()
    dinfer_engine.initialize()
    accelerator.wait_for_everyone()
    _log(accelerator, f"[Setup] dInfer engine initialized in {time.time() - t0:.1f}s")

    search_client = SearchClient(url=config.search.get("url", "http://127.0.0.1:8000/retrieve"),
                                 topk=int(config.search.get("topk", 3)))
    rollout_config = build_rollout_config(config)
    eval_rollout_config = build_eval_rollout_config(config)

    if _is_main(accelerator):
        health = config.search.get("url", "http://127.0.0.1:8000/retrieve").replace("/retrieve", "/docs")
        if not probe_url(health):
            wait_for_retriever_ready(RetrieverState(reused_existing=True), health_url=health, timeout_sec=300)
        ready = True
    else:
        ready = None
    if not _broadcast_object(ready):
        raise RuntimeError("Retriever not ready.")

    data = load_training_data(config)
    eval_data = load_eval_data(config)
    total_data = len(data)
    global_batch = max(1, int(config.rollout.num_task_per_step))
    k_sample = int(config.rollout.num_response_per_task)
    total_steps = int(config.experiment.total_step)
    trace_every = int(config.experiment.get("trace_every", 0) or 0)
    save_every = int(config.experiment.get("save_every", 0) or 0)
    checkpoint_warmup_steps = int(config.experiment.get("checkpoint_warmup_steps", 1) or 1)
    eval_k_sample = max(1, int(eval_cfg.get("num_response_per_task", 1) or 1))
    checkpoint_metric = str(eval_cfg.get("checkpoint_metric", "mean_f1") or "mean_f1")
    checkpoint_metric_mode = str(eval_cfg.get("checkpoint_metric_mode", "max") or "max")
    checkpoint_top_k = int(eval_cfg.get("checkpoint_top_k", 2) or 0)
    checkpoint_min_step = int(eval_cfg.get("checkpoint_min_step", 1) or 1)
    rng = random.Random(int(config.training.seed))
    if eval_before_step_every > 0 and not eval_data:
        _log(accelerator, "[Setup] eval_before_step_every > 0 but no eval dataset rows were loaded; skipping pre-step evals.")
    if save_every > 0 or checkpoint_warmup_steps > 1:
        _log(
            accelerator,
            "[Setup] experiment.save_every/checkpoint_warmup_steps are deprecated; "
            "checkpoint saving is now eval-driven via evaluation.checkpoint_*.",
        )
    if eval_before_step_every <= 0:
        _log(accelerator, "[Setup] pre-step eval disabled -> no RL checkpoints will be saved.")
    elif checkpoint_top_k <= 0:
        _log(accelerator, "[Setup] evaluation.checkpoint_top_k <= 0 -> eval will run but no checkpoints will be saved.")
    elif checkpoint_min_step > 1:
        _log(accelerator, f"[Setup] eval checkpoints only considered from step >= {checkpoint_min_step}.")
    if eval_before_step_every > 0:
        _log(
            accelerator,
            "[Setup] pre-step eval sampling: "
            f"temperature={eval_rollout_config.sampling.temperature} "
            f"top_p={eval_rollout_config.sampling.top_p} "
            f"top_k={eval_rollout_config.sampling.top_k} "
            f"confidence_threshold={eval_rollout_config.sampling.confidence_threshold}",
        )
        _log(accelerator, f"[Setup] dInfer runtime init threshold={runtime_init_threshold}")

    resident_dinfer = not bool(dinfer_cfg.get("unload_after_rollout", False))
    if resident_dinfer:
        _log(accelerator, "[Setup] resident layout: sharded GPU retriever + DP dInfer + FSDP learner")
        _record_dinfer_residency(project_dir=project_dir, stage="initialized", step=0,
                                 rank=rank, engine=dinfer_engine, learner=model)

    dinfer_weights_fresh = True  # engine just built from the SFT ckpt == learner weights

    for step in range(1, total_steps + 1):
        eval_metrics = None  # reset each outer step so stale eval is never logged
        # ---- sync engine weights from learner if stale ----
        if not dinfer_weights_fresh:
            _banner(accelerator, step, total_steps, "SYNC dInfer <- learner")
            _safe_cleanup(model, optimizer)
            if resident_dinfer:
                _record_dinfer_residency(project_dir=project_dir, stage="before_sync", step=step,
                                         rank=rank, engine=dinfer_engine, learner=model)
            t_sync = time.time()
            if bool(dinfer_cfg.get("streamed_gpu_sync", True)):
                from resident_sync import sync_fsdp_to_resident
                sync_metrics = sync_fsdp_to_resident(model, dinfer_engine,
                    verify=bool(dinfer_cfg.get("verify_streamed_sync", False)))
                _log(accelerator, f"[Sync] {sync_metrics}")
            else:
                train_state = _get_train_state_dict_for_runtime_sync(accelerator, model)
                dinfer_engine.update_from_train_state_dict(train_state)
                del train_state
            accelerator.wait_for_everyone()
            _log(accelerator, f"  >> sync done in {time.time() - t_sync:.1f}s")
            dinfer_weights_fresh = True
            if resident_dinfer:
                _record_dinfer_residency(project_dir=project_dir, stage="after_sync", step=step,
                                         rank=rank, engine=dinfer_engine, learner=model)

        # ---- optional eval before the training rollout ----
        if eval_before_step_every > 0 and step % eval_before_step_every == 0:
            eval_metrics = _run_pre_step_eval(
                accelerator,
                project_dir=project_dir,
                step=step,
                total_steps=total_steps,
                eval_items=eval_data,
                eval_k_sample=eval_k_sample,
                dinfer_engine=dinfer_engine,
                tokenizer=tokenizer,
                search_client=search_client,
                rollout_config=eval_rollout_config,
                world_size=world_size,
                rank=rank,
            )
            _maybe_save_best_eval_checkpoint(
                accelerator,
                model=model,
                tokenizer=tokenizer,
                eval_engine=dinfer_engine,
                project_dir=project_dir,
                step=step,
                source_model_path=pretrained_model,
                metrics_summary=eval_metrics,
                metric_name=checkpoint_metric,
                metric_mode=checkpoint_metric_mode,
                top_k=checkpoint_top_k,
                min_step=checkpoint_min_step,
            )

        # ---- select prompts (rank 0 picks, broadcast) ----
        start_idx = ((step - 1) * global_batch) % max(total_data, 1)
        if _is_main(accelerator):
            if start_idx + global_batch <= total_data:
                prompts = data[start_idx : start_idx + global_batch]
            else:
                prompts = data[start_idx:] + data[: global_batch - (total_data - start_idx)]
        else:
            prompts = None
        prompts = _broadcast_object(prompts)

        # ---- rollout (DP across ranks) ----
        _banner(accelerator, step, total_steps, "ROLLOUT",
                f"{global_batch} prompts x {k_sample} samples across {world_size} ranks")
        t_roll = time.time()
        episodes = run_rollouts_distributed(
            accelerator, prompts=prompts, k_sample=k_sample, dinfer_engine=dinfer_engine,
            tokenizer=tokenizer, search_client=search_client, rollout_config=rollout_config,
            world_size=world_size, rank=rank, project_dir=project_dir, step=step, gather_tag="train",
        )
        accelerator.wait_for_everyone()
        rollout_elapsed_s = time.time() - t_roll
        _log(accelerator, f"  >> rollout done in {rollout_elapsed_s:.0f}s ({len(episodes)} episodes)")

        # ---- reward + format guard (deterministic, identical on all ranks) ----
        _banner(accelerator, step, total_steps, "REWARD + format guard")
        summary = RFG.compute_group_rewards(
            episodes,
            gate_clean=float(config.reward.get("gate_clean", 1.0)),
            gate_soft=float(config.reward.get("gate_soft", 0.5)),
            gate_hard=float(config.reward.get("gate_hard", 0.0)),
        )
        _log(accelerator,
             f"  >> F1={summary['mean_f1']:.3f} reward={summary['mean_reward']:.3f} | "
             f"clean={summary['clean_format_frac']:.2f} soft={summary['soft_format_frac']:.2f} "
             f"hard={summary['hard_format_frac']:.2f} | upd={summary['mean_update_count']:.2f} "
             f"replan={summary['mean_replan_count']:.2f} collapse={summary['protocol_collapse_frac']:.2f}")
        if trace_every > 0 and step % trace_every == 0:
            _save_training_traces(
                accelerator,
                project_dir=project_dir,
                step=step,
                episodes=episodes,
                summary=summary,
            )

        # ---- build rows ----
        action_rows = build_all_rows(episodes, config=config, rng=rng)
        _log(accelerator, f"  >> built {len(action_rows)} action rows")

        # ---- offload dInfer engine weights to free GPU for training ----
        # (matches run_fused_rl.sh DINFER_UNLOAD_AFTER_ROLLOUT). The sync-before-rollout
        # step next iteration reloads the engine from the updated learner weights.
        if bool(dinfer_cfg.get("unload_after_rollout", False)):
            _banner(accelerator, step, total_steps, "OFFLOAD dInfer engine")
            _safe_cleanup(model, optimizer)
            t_offload = time.time()
            verbose_offload = os.environ.get("RL_VERBOSE_OFFLOAD", "0").strip().lower() in {
                "1", "true", "yes", "on"
            }
            with _suppress_output(not verbose_offload):
                dinfer_engine.unload_weights_keep_engine()
            accelerator.wait_for_everyone()
            _log(accelerator, f"  >> offload done in {time.time() - t_offload:.1f}s")
            dinfer_weights_fresh = False  # engine weights are now offloaded -> must sync before next rollout

        if resident_dinfer:
            _record_dinfer_residency(project_dir=project_dir, stage="before_ppo", step=step,
                                     rank=rank, engine=dinfer_engine, learner=model)

        # ---- reference logp + train ----
        _banner(accelerator, step, total_steps, "REF LOGP + PPO TRAIN", "action + edit")
        t_train = time.time()
        _log(accelerator, "  >> computing frozen reference log probabilities before PPO updates")
        attention_context = contextlib.nullcontext()
        if config.training.get("sdpa_backend") == "flash_cudnn":
            from torch.nn.attention import sdpa_kernel, SDPBackend
            attention_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION])
        with attention_context:
            sft_anchor.begin_step(episodes, step=step, learner=model, tokenizer=tokenizer,
                                  mask_id=mask_id, pad_id=pad_id)
            am = train_action_rows(
                accelerator, model=model, optimizer=optimizer, rows=action_rows, tokenizer=tokenizer,
                config=config, mask_id=mask_id, pad_id=pad_id, world_size=world_size, rank=rank,
                sft_anchor=sft_anchor,
            )
        accelerator.wait_for_everyone()
        dinfer_weights_fresh = False  # learner moved; engine now stale
        if resident_dinfer:
            _record_dinfer_residency(project_dir=project_dir, stage="after_ppo", step=step,
                                     rank=rank, engine=dinfer_engine, learner=model)

        _banner(accelerator, step, total_steps, "STEP DONE",
                f"rollout={rollout_elapsed_s:.0f}s train={time.time() - t_train:.0f}s")
        if _is_main(accelerator):
            print(
                f"  >> F1={summary['mean_f1']:.3f} reward={summary['mean_reward']:.3f} | "
                f"clean={summary['clean_format_frac']:.2f} soft={summary['soft_format_frac']:.2f} "
                f"hard={summary['hard_format_frac']:.2f} | "
                f"upd={summary['mean_update_count']:.2f} replan={summary['mean_replan_count']:.2f} "
                f"collapse={summary['protocol_collapse_frac']:.2f} | "
                f"action: rows={am['action_rows']:.0f} updates={am['action_updates']:.0f} "
                f"pl={am['action_policy_loss']:.4f} kl={am['action_kl_loss']:.4f}",
                flush=True,
            )

        # ---- wandb metrics for this outer step ----
        _wandb_payload = {
            "rollout/mean_f1": float(summary.get("mean_f1", 0.0) or 0.0),
            "rollout/mean_gated_score": float(summary.get("mean_gated_score", 0.0) or 0.0),
            "rollout/mean_reward": float(summary.get("mean_reward", 0.0) or 0.0),
            "rollout/clean_frac": float(summary.get("clean_format_frac", 0.0) or 0.0),
            "rollout/soft_frac": float(summary.get("soft_format_frac", 0.0) or 0.0),
            "rollout/hard_frac": float(summary.get("hard_format_frac", 0.0) or 0.0),
            "rollout/mean_update_count": float(summary.get("mean_update_count", 0.0) or 0.0),
            "rollout/mean_replan_count": float(summary.get("mean_replan_count", 0.0) or 0.0),
            "rollout/mean_search_count": float(summary.get("mean_search_count", 0.0) or 0.0),
            "rollout/mean_retrieval_used": float(summary.get("mean_retrieval_used", 0.0) or 0.0),
            "rollout/protocol_collapse_frac": float(summary.get("protocol_collapse_frac", 0.0) or 0.0),
            "rollout/num_episodes": float(summary.get("num_episodes", 0) or 0),
            "train/action_policy_loss": float(am.get("action_policy_loss", 0.0) or 0.0),
            "train/action_kl_loss": float(am.get("action_kl_loss", 0.0) or 0.0),
            "train/action_rows": float(am.get("action_rows", 0.0) or 0.0),
            "train/action_updates": float(am.get("action_updates", 0.0) or 0.0),
            "time/rollout_s": float(rollout_elapsed_s),
            "time/train_s": float(time.time() - t_train),
        }
        if eval_metrics:
            _add_eval_wandb_metrics(_wandb_payload, eval_metrics)
        _wandb_payload.update({f"train/{key}": value for key, value in am.items()
                               if key.startswith("sft_anchor_")})
        _wandb_log(accelerator, _wandb_payload, step)

    if wandb_enabled and _is_main(accelerator):
        try:
            import wandb

            if getattr(wandb, "run", None) is not None:
                wandb.finish()
        except Exception:  # noqa: BLE001 - shutdown must never fail the run
            pass
    _log(accelerator, "[Done] training complete.")


def _safe_cleanup(model, optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_checkpoint(
    accelerator,
    model,
    tokenizer,
    eval_engine,
    project_dir: Path,
    step,
    source_model_path,
    *,
    eval_runtime_dtype: Optional[torch.dtype] = None,
) -> None:
    save_path = project_dir / "ckpt" / f"step_{step}"
    state_dict = _get_train_state_dict_for_runtime_sync(accelerator, model, rank0_only=True)
    if accelerator.is_main_process:
        state_dict = _cast_state_dict_for_eval_runtime(
            state_dict,
            eval_engine=eval_engine,
            fallback_dtype=eval_runtime_dtype,
        )
        if any(name.endswith("mlp.w1") for name in state_dict):
            from fused_experts import unfuse_checkpoint_state
            state_dict = unfuse_checkpoint_state(state_dict)
        save_path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(
            save_path, is_main_process=True, save_function=accelerator.save, state_dict=state_dict,
        )
        _patch_saved_config_dtype(save_path, eval_runtime_dtype)
        tokenizer.save_pretrained(save_path)
        # copy modeling code for trust_remote_code
        import glob
        import shutil

        for src in glob.glob(os.path.join(source_model_path, "*.py")):
            dst = save_path / os.path.basename(src)
            if not dst.exists():
                shutil.copy2(src, dst)
        dtype_name = _torch_dtype_config_name(eval_runtime_dtype) or "original"
        print(f"[save] checkpoint -> {save_path} (runtime_dtype={dtype_name})", flush=True)
    del state_dict
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
