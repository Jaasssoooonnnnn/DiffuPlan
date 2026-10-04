"""BF16 MoE tile configuration for B200 GPUs."""

from pathlib import Path
import os
import torch


def install(engine):
    if not getattr(engine, "b200_moe_tiles", False):
        return
    import triton

    if (
        triton.__version__ != "3.4.0"
        or torch.cuda.get_device_name(engine.device) != "NVIDIA B200"
    ):
        return
    # Respect an explicitly supplied kernel table. No runtime autotuning is run.
    if "SGLANG_MOE_CONFIG_DIR" in os.environ:
        return
    os.environ["SGLANG_MOE_CONFIG_DIR"] = str(Path(__file__).parent / "moe_configs")
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
        get_moe_configs,
    )

    get_moe_configs.cache_clear()
    print(f"[dInfer][rank{engine.rank}] fixed B200 BF16 MoE tiles enabled", flush=True)
