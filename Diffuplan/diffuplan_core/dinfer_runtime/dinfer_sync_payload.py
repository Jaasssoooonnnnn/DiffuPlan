from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def _trace_rl_normalize_train_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    normalized_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        norm_key = key[7:] if key.startswith("module.") else key
        if isinstance(value, torch.nn.Parameter):
            value = value.data
        normalized_state_dict[norm_key] = value.detach().cpu()
    return normalized_state_dict


def _trace_rl_tp_split(
    tensor: torch.Tensor,
    *,
    dim: int,
    rank: int,
    world: int,
    is_w13: bool = False,
) -> torch.Tensor:
    if world <= 1:
        return tensor
    if is_w13:
        shard_size = tensor.size(dim) // 2
        size = shard_size // world
        w1 = tensor.narrow(dim, rank * size, size)
        w3 = tensor.narrow(dim, shard_size + rank * size, size)
        return torch.cat([w1, w3], dim=dim)
    size = tensor.size(dim) // world
    return tensor.narrow(dim, rank * size, size)


def _trace_rl_resolve_rank_layout(
    *,
    target_rank: int,
    target_world_size: int,
    parallel_mode: str,
) -> Tuple[int, int, int, int, int, int]:
    resolved_mode = str(parallel_mode).strip().lower()
    world_size = max(1, int(target_world_size))
    rank = max(0, int(target_rank))
    if resolved_mode == "dp":
        return 0, 1, 0, 1, 0, 1
    if resolved_mode == "tp":
        return rank, world_size, rank, world_size, rank, world_size
    raise ValueError(f"Unsupported sync parallel_mode={parallel_mode!r}")


def _trace_rl_load_expert_layout(
    *,
    model_config: Any,
    expert_map_path: str,
    ep_rank: int,
    ep_size: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    num_experts = int(model_config.num_experts)
    num_layers = int(model_config.num_hidden_layers)
    if num_layers == 20:
        expert_map_file = Path(expert_map_path) / f"mini_expert_map_{ep_size}.pt"
    else:
        expert_map_file = Path(expert_map_path) / f"flash_expert_map_{ep_size}.pt"

    if expert_map_file.exists():
        expert_map = torch.load(expert_map_file, map_location="cpu")
    else:
        expert_map = torch.zeros((num_layers, num_experts), dtype=torch.int32)
        experts_per_rank = max(1, num_experts // max(1, ep_size))
        for expert_id in range(num_experts):
            expert_map[:, expert_id] = min(ep_size - 1, expert_id // experts_per_rank)

    arange_n = torch.arange(num_experts, dtype=torch.int64)
    per_gpu_expert_mapping = [arange_n[expert_map[layer_id] == ep_rank] for layer_id in range(num_layers)]
    per_gpu_inverse_mapping = [
        torch.ones(num_experts, dtype=torch.int64).mul(-1) for _ in range(num_layers)
    ]
    for layer_id in range(num_layers):
        per_gpu_inverse_mapping[layer_id][per_gpu_expert_mapping[layer_id]] = torch.arange(
            per_gpu_expert_mapping[layer_id].shape[0],
            dtype=torch.int64,
        )
    return per_gpu_expert_mapping, per_gpu_inverse_mapping


def trace_rl_prepare_rank_local_sync_payload(
    state_dict: Dict[str, torch.Tensor],
    *,
    model_config: Any,
    expert_map_path: str,
    target_rank: int,
    target_world_size: int,
    parallel_mode: str = "dp",
) -> Dict[str, Any]:
    normalized_state_dict = _trace_rl_normalize_train_state_dict(state_dict)
    tp_rank, tp_size, moe_tp_rank, moe_tp_size, ep_rank, ep_size = _trace_rl_resolve_rank_layout(
        target_rank=target_rank,
        target_world_size=target_world_size,
        parallel_mode=parallel_mode,
    )
    per_gpu_expert_mapping, per_gpu_inverse_mapping = _trace_rl_load_expert_layout(
        model_config=model_config,
        expert_map_path=expert_map_path,
        ep_rank=ep_rank,
        ep_size=ep_size,
    )

    num_layers = int(model_config.num_hidden_layers)
    new_state_dict: Dict[str, torch.Tensor] = {}
    gate_projs = [{} for _ in range(num_layers)]
    up_projs = [{} for _ in range(num_layers)]
    down_projs = [{} for _ in range(num_layers)]

    for key, value in normalized_state_dict.items():
        if ".mlp.experts." in key:
            layer_id = int(key.split(".mlp.experts.")[0].split(".")[-1])
            expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
            if layer_id < num_layers:
                if "gate_proj" in key:
                    gate_projs[layer_id][expert_id] = value
                elif "up_proj" in key:
                    up_projs[layer_id][expert_id] = value
                elif "down_proj" in key:
                    down_projs[layer_id][expert_id] = value
        else:
            new_state_dict[key] = value

    for layer_id in range(num_layers):
        if f"model.layers.{layer_id}.mlp.w1" in normalized_state_dict:
            new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = _trace_rl_tp_split(
                normalized_state_dict[f"model.layers.{layer_id}.mlp.w1"][
                    per_gpu_expert_mapping[layer_id]
                ],
                dim=1,
                rank=moe_tp_rank,
                world=moe_tp_size,
                is_w13=True,
            ).contiguous()
            new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = _trace_rl_tp_split(
                normalized_state_dict[f"model.layers.{layer_id}.mlp.w2"][
                    per_gpu_expert_mapping[layer_id]
                ],
                dim=2,
                rank=moe_tp_rank,
                world=moe_tp_size,
            ).contiguous()
            new_state_dict.pop(f"model.layers.{layer_id}.mlp.w1", None)
            new_state_dict.pop(f"model.layers.{layer_id}.mlp.w2", None)

        if gate_projs[layer_id]:
            w13_weight = []
            w2_weight = []
            for expert_id in per_gpu_expert_mapping[layer_id]:
                expert_idx = int(expert_id)
                gate_proj = gate_projs[layer_id][expert_idx]
                up_proj = up_projs[layer_id][expert_idx]
                down_proj = down_projs[layer_id][expert_idx]
                w13_weight.append(torch.cat([gate_proj, up_proj], dim=0))
                w2_weight.append(down_proj)
            new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = _trace_rl_tp_split(
                torch.stack(w13_weight, dim=0),
                dim=1,
                rank=moe_tp_rank,
                world=moe_tp_size,
                is_w13=True,
            ).contiguous()
            new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = _trace_rl_tp_split(
                torch.stack(w2_weight, dim=0),
                dim=2,
                rank=moe_tp_rank,
                world=moe_tp_size,
            ).contiguous()

        if f"model.layers.{layer_id}.mlp.gate.expert_bias" in normalized_state_dict:
            new_state_dict[f"model.layers.{layer_id}.mlp.correction_bias"] = normalized_state_dict[
                f"model.layers.{layer_id}.mlp.gate.expert_bias"
            ]
            new_state_dict.pop(f"model.layers.{layer_id}.mlp.gate.expert_bias", None)

        if f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight" in normalized_state_dict:
            part_size = (
                normalized_state_dict[
                    f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
                ].shape[0]
                // tp_size
            )
            part_start = tp_rank * part_size
            part_end = (tp_rank + 1) * part_size
            new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_up_proj.weight"] = torch.cat(
                [
                    normalized_state_dict[
                        f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
                    ][part_start:part_end],
                    normalized_state_dict[
                        f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"
                    ][part_start:part_end],
                ],
                dim=0,
            ).contiguous()
            new_state_dict.pop(f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight", None)
            new_state_dict.pop(f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight", None)

        if f"model.layers.{layer_id}.mlp.gate_proj.weight" in normalized_state_dict:
            part_size = normalized_state_dict[
                f"model.layers.{layer_id}.mlp.gate_proj.weight"
            ].shape[0] // tp_size
            part_start = tp_rank * part_size
            part_end = (tp_rank + 1) * part_size
            new_state_dict[f"model.layers.{layer_id}.mlp.gate_up_proj.weight"] = torch.cat(
                [
                    normalized_state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"][
                        part_start:part_end
                    ],
                    normalized_state_dict[f"model.layers.{layer_id}.mlp.up_proj.weight"][
                        part_start:part_end
                    ],
                ],
                dim=0,
            ).contiguous()
            new_state_dict.pop(f"model.layers.{layer_id}.mlp.gate_proj.weight", None)
            new_state_dict.pop(f"model.layers.{layer_id}.mlp.up_proj.weight", None)

    if "model.word_embeddings.weight" in normalized_state_dict:
        new_state_dict["model.full_word_embeddings.weight"] = normalized_state_dict[
            "model.word_embeddings.weight"
        ]

    payload_bytes = 0
    for value in new_state_dict.values():
        payload_bytes += int(value.numel()) * int(value.element_size())

    return {
        "weights": new_state_dict,
        "expert_map_cpu_by_layer": [tensor.clone() for tensor in per_gpu_inverse_mapping],
        "target_rank": int(target_rank),
        "target_world_size": int(target_world_size),
        "parallel_mode": str(parallel_mode).strip().lower(),
        "payload_bytes": int(payload_bytes),
    }
