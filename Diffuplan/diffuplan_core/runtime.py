"""Runtime utilities for local LLaDA2 inference jobs."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen


@dataclass
class RetrieverState:
    process: object | None = None
    log_path: str = ""
    reused_existing: bool = False


DEFAULT_RETRIEVER_PYTHON = sys.executable


def ensure_runtime_env(cache_root: Optional[str | Path] = None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("DINFER_DISABLE_COORDINATE_DESCENT_TUNING", "1")

    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if not conda_prefix:
        conda_prefix = str(Path(sys.executable).parents[1].resolve())
        os.environ.setdefault("CONDA_PREFIX", conda_prefix)
    if conda_prefix:
        conda_root = Path(conda_prefix).resolve()
        cuda_lib_dirs = [
            conda_root / "lib",
            conda_root / "targets" / "x86_64-linux" / "lib",
            conda_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia" / "cuda_runtime" / "lib",
        ]
        if not os.environ.get("CUDA_HOME") and not os.environ.get("CUDA_PATH"):
            if any((path / "libcudart.so.12").exists() for path in cuda_lib_dirs):
                os.environ["CUDA_HOME"] = str(conda_root)

        existing_ld = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
        prepend_ld = [str(path) for path in cuda_lib_dirs if path.exists()]
        merged_ld = []
        for item in prepend_ld + existing_ld:
            if item and item not in merged_ld:
                merged_ld.append(item)
        if merged_ld:
            os.environ["LD_LIBRARY_PATH"] = ":".join(merged_ld)

        os.environ["LIBRARY_PATH"] = f"{conda_prefix}/lib:{os.environ.get('LIBRARY_PATH', '')}".rstrip(":")
        os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        os.environ["CPATH"] = f"{conda_prefix}/include:{os.environ.get('CPATH', '')}".rstrip(":")

    base = Path(cache_root or os.environ.get("SLURM_TMPDIR", f"/tmp/{os.environ.get('USER', 'user')}"))
    for name in ("triton_cache", "torchinductor_cache", "torch_extensions"):
        path = base / name
        path.mkdir(parents=True, exist_ok=True)
        env_name = {
            "triton_cache": "TRITON_CACHE_DIR",
            "torchinductor_cache": "TORCHINDUCTOR_CACHE_DIR",
            "torch_extensions": "TORCH_EXTENSIONS_DIR",
        }[name]
        os.environ.setdefault(env_name, str(path))


def detect_visible_gpu_ids() -> list[str]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    if cuda_visible_devices and cuda_visible_devices not in {"-1", "NoDevFiles"}:
        return [str(idx) for idx, item in enumerate(cuda_visible_devices.split(",")) if item]

    if which("nvidia-smi") is not None:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        count = sum(1 for line in proc.stdout.splitlines() if line.startswith("GPU "))
        if count > 0:
            return [str(idx) for idx in range(count)]

    try:
        import torch

        count = int(torch.cuda.device_count())
    except Exception:
        count = 0
    return [str(idx) for idx in range(count)]


def which(cmd: str) -> Optional[str]:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / cmd
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def probe_url(url: str, timeout_sec: float = 2.0) -> bool:
    try:
        with urlopen(url, timeout=timeout_sec) as response:
            return 200 <= int(response.status) < 400
    except (URLError, TimeoutError, ValueError, OSError):
        return False


def launch_retriever_async(
    *,
    retriever_root: str | Path,
    health_url: str = "http://127.0.0.1:8000/docs",
    log_dir: str | Path,
    topk: int = 3,
    gpu_ids: Optional[list[str]] = None,
    gpu_shard: bool = True,
    gpu_load_mode: str = "batch",
) -> RetrieverState:
    if probe_url(health_url):
        return RetrieverState(reused_existing=True)

    root = Path(retriever_root).expanduser().resolve()
    launch_script = root / "launch_retriever.sh"
    if not launch_script.exists():
        raise FileNotFoundError(f"launch_retriever.sh not found: {launch_script}")

    resolved_gpu_ids = gpu_ids if gpu_ids is not None else detect_visible_gpu_ids()
    if not resolved_gpu_ids:
        raise RuntimeError("No visible GPUs found for retriever launch.")

    log_root = Path(log_dir).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"retriever_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    env = os.environ.copy()
    env["RETRIEVER_USE_GPU"] = "1"
    env["RETRIEVER_GPU_IDS"] = ",".join(resolved_gpu_ids)
    env["RETRIEVER_GPU_SHARD"] = "1" if gpu_shard else "0"
    env["RETRIEVER_GPU_LOAD_MODE"] = gpu_load_mode
    env["RETRIEVER_TOPK"] = str(int(topk))
    env.setdefault("RETRIEVER_PYTHON_EXEC", resolve_retriever_python())
    env.setdefault("RETRIEVER_FORCE_CPU", "0")

    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            ["bash", str(launch_script)],
            cwd=str(root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return RetrieverState(process=process, log_path=str(log_path), reused_existing=False)


def wait_for_retriever_ready(
    state: RetrieverState,
    *,
    health_url: str = "http://127.0.0.1:8000/docs",
    timeout_sec: int = 180,
) -> None:
    deadline = time.time() + max(1, int(timeout_sec))
    while time.time() < deadline:
        if probe_url(health_url):
            return
        if state.process is not None and state.process.poll() is not None:
            raise RuntimeError(
                f"Retriever exited early with code {state.process.returncode}. "
                f"Check log: {state.log_path}"
            )
        time.sleep(1)
    raise TimeoutError(f"Retriever did not become ready within {timeout_sec}s. Log: {state.log_path}")


def resolve_retriever_python() -> str:
    configured = os.environ.get("RETRIEVER_PYTHON_EXEC", "").strip()
    if configured:
        return configured
    if Path(DEFAULT_RETRIEVER_PYTHON).exists():
        return DEFAULT_RETRIEVER_PYTHON
    return sys.executable
