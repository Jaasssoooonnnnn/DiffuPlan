import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Literal, Tuple, Optional

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.models.registry import ModelRegistry
ModelRegistry.register_modeling_path("models.llada2_moe")
from dataset.data_transform import (
    process_diffuplan_sft_example,
    process_mdm_sft_example,
    process_mdm_tokenized_example,
)
from dataset.diffuplan_joint_loss import compute_diffuplan_edit_joint_marginal_loss
from dataset import build_local_dataset


logger = helper.create_logger(__name__)

@dataclass
class LLaDA2ModelArguments(ModelArguments):
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = field(
        default="sdpa",
        metadata={"help": "Attention implementation to use."},
    )


@dataclass
class LLaDA2DataArguments(DataArguments):
    data_type: Literal["conversation", "tokenid", "diffuplan"] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
    datasets_type: Literal["mapping", "local"] = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    text_keys: str = field(
        default="messages",
        metadata={"help": "Key to get text from the training data."},
    )
    noise_range_low: float = field(
        default=0.3,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"}
    )
    noise_range_high: float = field(
        default=0.8,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.noise_range_low > self.noise_range_high:
            raise ValueError(
                f"noise_range_low ({self.noise_range_low}) "
                f"cannot be greater than noise_range_high ({self.noise_range_high})."
            )

        if not (0.0 <= self.noise_range_low <= 1.0):
            raise ValueError(
                f"noise_range_low must be between 0.0 and 1.0, but got {self.noise_range_low}."
            )

        if not (0.0 <= self.noise_range_high <= 1.0):
            raise ValueError(
                f"noise_range_high must be between 0.0 and 1.0, but got {self.noise_range_high}."
            )


@dataclass
class LLaDA2TrainingArguments(TrainingArguments):
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW optimizer beta1."},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "AdamW optimizer beta2"},
    )
    block_diffusion_mode: bool = field(
        default=False,
        metadata={"help": "If train MDM in block_diffusion mode. True: use block_diffusion, False: full_attention"}
    )
    block_size: int = field(
        default=32,
        metadata={"help": "The block size for block diffusion block size"}
    )
    same_token_labels: bool = field(
        default=False,
        metadata={"help": "If use same token location labels. True: no shift, False: use next-token prediction shift."}
    )
    save_unfused_hf_weights: bool = field(
        default=False,
        metadata={"help": "Convert fused LLaDA2 HF checkpoint to unfused HF layout after saving."},
    )
    keep_fused_hf_weights: bool = field(
        default=True,
        metadata={"help": "Keep hf_ckpt after writing hf_ckpt_unfused."},
    )
    delete_distributed_checkpoint_after_hf: bool = field(
        default=False,
        metadata={"help": "Delete DCP .distcp/.metadata/extra_state files after HF weights are saved."},
    )


@dataclass
class Arguments:
    model: "LLaDA2ModelArguments" = field(default_factory=LLaDA2ModelArguments)
    data: "LLaDA2DataArguments" = field(default_factory=LLaDA2DataArguments)
    train: "LLaDA2TrainingArguments" = field(default_factory=LLaDA2TrainingArguments)


def block_diffusion_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices
    block_q = torch.where(x0_flag_q == 1,
                          (q_idx - n) // block_size,
                          q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1,
                           (kv_idx - n) // block_size,
                           kv_idx // block_size)

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (
        (block_q > block_kv)
        & (x0_flag_kv == 1)
        & (x0_flag_q == 0)
    )

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal


def _build_diffuplan_full_inputs_and_mask(
    *,
    noisy_input_ids: torch.Tensor,
    clean_input_ids: torch.Tensor,
    real_token_attention_mask: Optional[torch.Tensor],
    edit_rows: torch.Tensor,
    block_diffusion_attn_mask_prototype: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a unified 2N batch while giving edit rows full-canvas noisy attention only."""

    batch_size, seq_len = noisy_input_ids.shape
    device = noisy_input_ids.device
    full_len = seq_len * 2

    pad_half = torch.full_like(clean_input_ids, int(pad_token_id))
    second_half = torch.where(edit_rows[:, None], pad_half, clean_input_ids)
    full_input_ids = torch.cat([noisy_input_ids, second_half], dim=1)

    half_position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
    position_ids = torch.cat([half_position_ids, half_position_ids], dim=0).unsqueeze(0).expand(
        batch_size, -1
    ).clone()

    attention_mask = block_diffusion_attn_mask_prototype.to(device=device).expand(
        batch_size, -1, -1, -1
    ).clone()

    if edit_rows.any():
        edit_indices = torch.nonzero(edit_rows, as_tuple=False).flatten()
        if real_token_attention_mask is None:
            real_token_mask = torch.ones_like(noisy_input_ids, dtype=torch.bool)
        else:
            real_token_mask = real_token_attention_mask.to(device=device, dtype=torch.bool)
        edit_real_token_mask = real_token_mask.index_select(0, edit_indices)

        edit_allowed = torch.zeros(
            (edit_indices.numel(), full_len, full_len),
            dtype=torch.bool,
            device=device,
        )
        edit_allowed[:, :seq_len, :seq_len] = edit_real_token_mask[:, None, :]
        second_half_idx = torch.arange(seq_len, full_len, device=device)
        edit_allowed[:, second_half_idx, second_half_idx] = True

        edit_attention_mask = torch.zeros(
            (edit_indices.numel(), 1, full_len, full_len),
            dtype=attention_mask.dtype,
            device=device,
        )
        edit_attention_mask.masked_fill_(edit_allowed[:, None].logical_not(), float("-inf"))
        attention_mask.index_copy_(0, edit_indices, edit_attention_mask)

    return full_input_ids, position_ids, attention_mask


def convert_fused_hf_to_unfused(hf_weights_path: str) -> str:
    src = Path(hf_weights_path)
    dst = src.with_name("hf_ckpt_unfused")
    converter = Path(__file__).resolve().parents[2] / "tools" / "convert_llada2_fused_to_unfused.py"
    if not converter.exists():
        raise FileNotFoundError(f"Converter not found: {converter}")
    subprocess.run(
        [sys.executable, str(converter), "--src", str(src), "--dst", str(dst), "--overwrite"],
        check=True,
    )
    if not (dst / "model.safetensors.index.json").exists():
        raise FileNotFoundError(f"Unfused conversion did not create model.safetensors.index.json: {dst}")
    return str(dst)


def delete_distributed_checkpoint_payload(save_checkpoint_path: str) -> None:
    ckpt_dir = Path(save_checkpoint_path)
    for item in ckpt_dir.glob("*.distcp"):
        item.unlink()
    metadata = ckpt_dir / ".metadata"
    if metadata.exists():
        metadata.unlink()
    extra_state = ckpt_dir / "extra_state"
    if extra_state.exists():
        shutil.rmtree(extra_state)


def main():
    dist.init_process_group(backend=get_nccl_backend())
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if args.data.data_type == "conversation":
        if not tokenizer.chat_template:
            raise ValueError(f"No chat template found in the tokenizer.")

        transform = partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=156895, 
        )
    elif args.data.data_type == "tokenid":
        transform = partial(
            process_mdm_tokenized_example,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=156895, 
        )
    elif args.data.data_type == "diffuplan":
        transform = partial(
            process_diffuplan_sft_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=156895,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
        elif args.data.datasets_type == "local":
            logger.info_rank0("Start building local dataset")
            train_dataset = build_local_dataset(args.data.train_path, transform=transform)
        
        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type == "mapping" or args.data.datasets_type == "local":
            dataset_length = dataset_length / args.train.data_parallel_size
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        betas=(args.train.beta1, args.train.beta2),
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )

    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        # save model_assets before training
        model_assets = [model_config, tokenizer]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    # Build block diffusion attention mask
    if args.train.block_diffusion_mode:
        bd_attn_full_len = args.data.max_seq_len * 2
        block_size = args.train.block_size
        # NOTE: Boolean dtype block diffusion attention mask
        block_diffusion_attn_mask_flag = block_diffusion_mask(
            b=None, h=None,
            q_idx=torch.arange(bd_attn_full_len)[:, None],
            kv_idx=torch.arange(bd_attn_full_len)[None, :],
            block_size=block_size,
            n=args.data.max_seq_len
        ).unsqueeze(0).unsqueeze(0)
        
        block_diffusion_attn_mask_prototype = torch.zeros_like(
            block_diffusion_attn_mask_flag, 
            dtype=torch.float32 if args.train.enable_mixed_precision else torch.bfloat16
        )
        block_diffusion_attn_mask_prototype.masked_fill_(block_diffusion_attn_mask_flag.logical_not(), float("-inf"))

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            synchronize()
            start_time = time.time()
            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }

                clean_input_ids = micro_batch.pop("input_ids")
                noisy_input_ids = micro_batch.pop("noisy_input_ids")
                real_token_attention_mask = micro_batch.pop("attention_mask", None)
                labels = micro_batch.pop("labels", None)
                diffuplan_record_type = micro_batch.pop("diffuplan_record_type", None)
                diffuplan_edit_loss_mask = micro_batch.pop("diffuplan_edit_loss_mask", None)
                diffuplan_edit_target_ids = micro_batch.pop("diffuplan_edit_target_ids", None)
                diffuplan_edit_target_len = micro_batch.pop("diffuplan_edit_target_len", None)
                diffuplan_edit_segment_kinds = micro_batch.pop("diffuplan_edit_segment_kinds", None)
                diffuplan_edit_segment_lens = micro_batch.pop("diffuplan_edit_segment_lens", None)

                with model_fwd_context:
                    batch_size = clean_input_ids.shape[0]
                    if diffuplan_record_type is None:
                        edit_rows = torch.zeros(batch_size, dtype=torch.bool, device=clean_input_ids.device)
                    else:
                        edit_rows = diffuplan_record_type.to(device=clean_input_ids.device).eq(1)
                    action_rows = edit_rows.logical_not()

                    if args.train.block_diffusion_mode:
                        pad_token_id = tokenizer.pad_token_id
                        if pad_token_id is None:
                            pad_token_id = getattr(model_config, "pad_token_id", 0)
                        full_input_ids, position_ids, attention_mask = _build_diffuplan_full_inputs_and_mask(
                            noisy_input_ids=noisy_input_ids,
                            clean_input_ids=clean_input_ids,
                            real_token_attention_mask=real_token_attention_mask,
                            edit_rows=edit_rows,
                            block_diffusion_attn_mask_prototype=block_diffusion_attn_mask_prototype,
                            pad_token_id=int(pad_token_id),
                        )
                        micro_batch["input_ids"] = full_input_ids
                        micro_batch["position_ids"] = position_ids
                        micro_batch["attention_mask"] = attention_mask
                    else:
                        micro_batch["input_ids"] = noisy_input_ids
                        micro_batch["attention_mask"] = None

                    logits: "torch.Tensor" = model(**micro_batch, use_cache=False, output_router_logits=False).logits
                    noisy_logits = logits[:, : noisy_input_ids.shape[1]].contiguous()

                    total_loss_sum = noisy_logits.new_zeros((), dtype=torch.float32)
                    total_token_count = torch.zeros((), dtype=torch.float32, device=noisy_logits.device)

                    if action_rows.any():
                        action_noisy_logits = noisy_logits[action_rows]
                        action_labels = labels[action_rows]
                        if args.train.same_token_labels:
                            unscaled_loss = torch.nn.functional.cross_entropy(
                                action_noisy_logits.view(-1, action_noisy_logits.shape[-1]),
                                action_labels.view(-1),
                                reduction="none",
                            )
                            total_loss_sum = total_loss_sum + unscaled_loss.sum()
                            total_token_count = total_token_count + (action_labels != -100).sum().float()
                        else:
                            shifted_noisy_logits = action_noisy_logits[:, :-1, :].contiguous()
                            shifted_labels = action_labels[:, 1:].contiguous()
                            unscaled_loss = torch.nn.functional.cross_entropy(
                                shifted_noisy_logits.view(-1, shifted_noisy_logits.shape[-1]),
                                shifted_labels.view(-1),
                                reduction="none",
                            )
                            total_loss_sum = total_loss_sum + unscaled_loss.sum()
                            total_token_count = total_token_count + (shifted_labels != -100).sum().float()

                    if edit_rows.any():
                        if not args.train.same_token_labels:
                            raise ValueError("Diffuplan edit joint marginal loss requires same_token_labels=true.")
                        joint_loss_sum, joint_token_count = compute_diffuplan_edit_joint_marginal_loss(
                            logits=noisy_logits,
                            edit_loss_mask=diffuplan_edit_loss_mask,
                            target_ids=diffuplan_edit_target_ids,
                            target_lens=diffuplan_edit_target_len,
                            segment_kinds=diffuplan_edit_segment_kinds,
                            segment_lens=diffuplan_edit_segment_lens,
                            record_types=diffuplan_record_type,
                            expand_token_id=156901,
                        )
                        total_loss_sum = total_loss_sum + joint_loss_sum
                        total_token_count = total_token_count + joint_token_count.float()

                    loss = total_loss_sum / total_token_count.clamp_min(1.0) / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            # Prefer model-provided clip_grad_norm_ (now both FSDP1 and FSDP2 registers custom grad norm clipping)
            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                logger.info_rank0(
                    "Can NOT find regitsered clip_grad_norm_ method in the model, using PyTorch default implementation.."
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)

                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
        if args.train.save_unfused_hf_weights:
            unfused_path = convert_fused_hf_to_unfused(hf_weights_path)
            logger.info_rank0(f"Unfused Huggingface checkpoint saved at {unfused_path} successfully!")
            if not args.train.keep_fused_hf_weights:
                shutil.rmtree(hf_weights_path)
                logger.info_rank0(f"Removed fused Huggingface checkpoint at {hf_weights_path}")
        if args.train.delete_distributed_checkpoint_after_hf:
            delete_distributed_checkpoint_payload(save_checkpoint_path)
            logger.info_rank0(f"Removed distributed checkpoint payload under {save_checkpoint_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
