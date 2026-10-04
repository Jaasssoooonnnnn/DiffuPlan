#!/usr/bin/env python3
"""Convert LLaDA2 fused MoE expert safetensors to the unfused HF layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


FUSED_SUFFIX_TO_UNFUSED = {
    "experts.down_proj": "experts.{expert}.down_proj.weight",
    "experts.gate_proj": "experts.{expert}.gate_proj.weight",
    "experts.up_proj": "experts.{expert}.up_proj.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split LLaDA2 fused MoE tensors like "
            "model.layers.N.mlp.experts.down_proj into per-expert tensors "
            "like model.layers.N.mlp.experts.0.down_proj.weight."
        )
    )
    parser.add_argument("--src", required=True, type=Path, help="Source fused HF checkpoint")
    parser.add_argument("--dst", required=True, type=Path, help="Destination unfused HF checkpoint")
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Optional base checkpoint to copy missing metadata files from, e.g. generation_config.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the destination first if it already exists",
    )
    return parser.parse_args()


def copy_sidecar_files(src: Path, dst: Path, base: Path | None) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    skip_names = {"model.safetensors.index.json", "__pycache__"}
    skip_suffixes = {".safetensors"}

    for item in src.iterdir():
        if item.name in skip_names:
            continue
        if item.is_file() and item.suffix in skip_suffixes:
            continue
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    if base is not None:
        for name in ("generation_config.json",):
            target = dst / name
            source = base / name
            if not target.exists() and source.exists():
                shutil.copy2(source, target)


def normalize_config(dst: Path) -> None:
    config_path = dst / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("_moe_implementation", None)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")


def split_fused_tensor(name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor] | None:
    for fused_suffix, unfused_pattern in FUSED_SUFFIX_TO_UNFUSED.items():
        marker = f".mlp.{fused_suffix}"
        if name.endswith(marker):
            prefix = name[: -len(fused_suffix)]
            return {
                prefix + unfused_pattern.format(expert=expert_idx): tensor[expert_idx].contiguous()
                for expert_idx in range(tensor.shape[0])
            }
    return None


def convert_shard(src_shard: Path, dst_shard: Path) -> dict[str, str]:
    print(f"[convert] {src_shard.name}", flush=True)
    source_tensors = load_file(src_shard, device="cpu")
    converted: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}

    for name, tensor in source_tensors.items():
        split = split_fused_tensor(name, tensor)
        if split is None:
            converted[name] = tensor
            weight_map[name] = dst_shard.name
        else:
            converted.update(split)
            for split_name in split:
                weight_map[split_name] = dst_shard.name

    save_file(converted, dst_shard, metadata={"format": "pt"})
    print(
        f"[convert] {src_shard.name}: {len(source_tensors)} source tensors -> "
        f"{len(converted)} destination tensors",
        flush=True,
    )
    del source_tensors
    del converted
    return weight_map


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    base = args.base.resolve() if args.base is not None else None

    if not (src / "model.safetensors.index.json").exists():
        raise FileNotFoundError(f"Missing source index: {src / 'model.safetensors.index.json'}")
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination exists: {dst}. Pass --overwrite to replace it.")
        shutil.rmtree(dst)

    copy_sidecar_files(src, dst, base)
    normalize_config(dst)

    src_index = json.loads((src / "model.safetensors.index.json").read_text())
    src_shards = sorted(set(src_index["weight_map"].values()))
    dst_weight_map: dict[str, str] = {}
    for shard_name in src_shards:
        dst_weight_map.update(convert_shard(src / shard_name, dst / shard_name))

    dst_index = {
        "metadata": src_index.get("metadata", {}),
        "weight_map": dict(sorted(dst_weight_map.items())),
    }
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(dst_index, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"[done] wrote {len(dst_weight_map)} tensors to {dst}", flush=True)


if __name__ == "__main__":
    main()
