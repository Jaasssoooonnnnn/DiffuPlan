"""Sampled-token SFT regularization on reconstructed TraceRL rollout slices.

The frozen reference covers all trajectory groups, including zero-advantage groups."""
from functools import partial
from copy import deepcopy
from pathlib import Path
import json
import random

import torch
import torch.distributed as dist

import losses as L
import trace_to_rows as TTR
from action_batching import collate_ragged, pack_ids


def fixed_sft_kl_sum(logp_new, logp_sft, mask, denominator, row_weight, *, tail=10.0):
    """Return a sum of length-normalized row K3 surrogates, independent of reward.

    k3(t) = exp(t)-1-t, t=log p_sft-log p_new. Beyond +/-tail use its tangent
    continuation. Unlike clamping only the exponential, this preserves the
    restoring gradient's sign for very large discrepancies and stays finite.
    """
    if tail <= 0:
        raise ValueError('KL tail threshold must be positive')
    weight = row_weight.detach().float()
    active = mask.bool() & weight[:, None].gt(0)
    t = torch.where(active, logp_sft.detach().float() - logp_new.float(), 0.)
    bounded = t.clamp(-tail, tail)
    slope = torch.expm1(bounded)
    k3 = slope - bounded + slope.detach() * (t - bounded)
    denom = torch.as_tensor(denominator, device=t.device).float().clamp_min(1)
    per_row = (k3 * active).sum(dim=1) / denom
    value = (per_row * weight).sum()
    return value, torch.stack(((t.abs().gt(tail) & active).sum(), active.sum())).detach()


def make_anchor_windows(built, *, updates, world_size, rank, pad_id, seed,
                        token_budget, max_rows, attention_budget):
    """One independent pass over all anchor rows, with equal-rank participation."""
    if not built or updates < 1:
        raise ValueError('Anchor needs replayable rows and at least one update')
    ids = list(range(len(built)))
    random.Random(seed).shuffle(ids)
    windows = []
    for update in range(updates):
        selected = ids[len(ids) * update // updates:len(ids) * (update + 1) // updates]
        packs = pack_ids(selected, built, token_budget=token_budget,
                         max_rows=max_rows, attention_budget=attention_budget)
        entries = [dict(row_indices=x, is_padding=False, pad_id=pad_id) for x in packs]
        if not entries:
            entries = [dict(row_indices=[ids[0]], is_padding=True, pad_id=pad_id)]
        while len(entries) % world_size:
            entries.append(dict(row_indices=[ids[0]], is_padding=True, pad_id=pad_id))
        windows.append(dict(batches=entries[rank::world_size], global_rows=len(selected)))
    return windows


def reference_mixed_precision(accelerator):
    """Keep buffer/input casting identical to the already prepared learner."""
    policy = accelerator.state.fsdp_plugin.mixed_precision_policy
    if policy is None:
        raise ValueError('Missing learner FSDP mixed-precision policy')
    return deepcopy(policy)


def load_frozen_sft(accelerator, config, checkpoint_has_qk_norm):
    """A separate BF16 FSDP model; never passed to the optimizer or weight sync."""
    import numpy as np
    from transformers import AutoConfig, AutoModelForCausalLM
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from learner_optimizations import install_learner_optimizations
    from fused_experts import fuse_expert_parameters
    from sm100_expert_gemm import install_sm100_expert_gemm

    checkpoint = str(config.sft_anchor.reference_model)
    if checkpoint != str(config.model.pretrained_model):
        raise ValueError('SFT anchor reference must match the policy initialization')
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[accelerator.device.index]):
            mc = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
            if not hasattr(mc, 'use_qk_norm'):
                mc.use_qk_norm = checkpoint_has_qk_norm(checkpoint)
            mc.fuse_cross_entropy, mc.use_cache = True, False
            reference = AutoModelForCausalLM.from_pretrained(
                checkpoint, config=mc, trust_remote_code=True,
                torch_dtype='auto', attn_implementation='sdpa')
            reference._rl_attention_dtype = torch.bfloat16
            install_learner_optimizations(reference, sparse_action_logits=True,
                grouped_reference_moe=True, logp_chunk_size=int(config.training.logp_chunk_size))
            if config.training.fused_moe_parameters:
                fuse_expert_parameters(reference)
            if config.training.sm100_expert_gemm:
                install_sm100_expert_gemm(reference)
            reference.requires_grad_(False)
            reference.eval()
            layer_type = type(reference.model.layers[0])
            reference = FSDP(reference,
                auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_type}),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=reference_mixed_precision(accelerator),
                device_id=accelerator.device, use_orig_params=True,
                sync_module_states=False, limit_all_gathers=True)
            reference.eval()
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return reference


class FrozenSFTAnchor:
    def __init__(self, accelerator, config, *, trainer, optimizer, reference_model=None):
        self.acc, self.config, self.trainer = accelerator, config, trainer
        self.cfg = config.sft_anchor
        self.coefficient = float(self.cfg.coefficient)
        if self.coefficient <= 0:
            raise ValueError('Enabled anchor needs a positive coefficient')
        self.updates = int(config.training.target_updates_per_outer_step)
        self.rank, self.world = accelerator.process_index, accelerator.num_processes
        self.device = accelerator.device
        self.reference = (reference_model if reference_model is not None else
            load_frozen_sft(accelerator, config, trainer._checkpoint_has_qk_norm))
        self.reference.requires_grad_(False)
        self.reference.eval()
        self.optimizer = optimizer
        self.rng = random.Random(int(config.training.seed) + int(self.cfg.seed_offset))
        self.assert_frozen()
        if self.acc.is_main_process:
            print(f'[SFT anchor] fixed={self.cfg.reference_model} coefficient={self.coefficient} '
                  f'zero-reward groups included; {self.updates} shared updates', flush=True)

    def assert_frozen(self):
        params = list(self.reference.parameters())
        if any(p.requires_grad or p.grad is not None for p in params):
            raise RuntimeError('Frozen SFT reference acquired gradients')
        ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        if any(id(p) in ids for p in params):
            raise RuntimeError('SFT reference is in the learner optimizer')

    def _batch(self, entry, *, with_reference=False):
        entry = dict(entry, pad_id=self.pad_id)
        batch = collate_ragged(self.built, entry, include_old_logp=False)
        if with_reference:
            ref = torch.zeros_like(batch['labels'], dtype=torch.float32)
            for i, row_id in enumerate(entry['row_indices']):
                row = self.built[row_id]
                if not entry['is_padding']:
                    cached = row['logp_sft']
                    ref[i, :len(cached)] = cached
            batch['logp_sft'] = ref
        return batch

    def _logp(self, model, batch):
        ext = batch['ext'].to(self.device)
        l0, l1 = batch['L0'].to(self.device), batch['L1'].to(self.device)
        att = L._action_attention(None, ext, self.pad_id, l0, l0, l1,
                                  int(self.config.training.block_size), torch.bfloat16)
        # The learner's Accelerate wrapper already enables autocast. The frozen
        # FSDP model has no such wrapper, so explicitly give both the same context.
        with self.acc.autocast():
            return L._action_token_logp(model, ext, att, batch['tok_idx'].to(self.device),
                batch['labels'].to(self.device), batch['pmask'].to(self.device), l0, l1,
                temperature=float(self.config.rollout.temperature),
                banned_token_ids=(self.mask_id, L.EXPAND_TOKEN_ID))

    @torch.no_grad()
    def _probe(self, model):
        training = model.training
        model.eval()
        result = self._logp(model, self._batch(self.probe_entry)).detach().float().cpu()
        model.train(training)
        return result

    def _max_across_ranks(self, value):
        tensor = torch.tensor(float(value), device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor)

    def begin_step(self, episodes, *, step, learner, tokenizer, mask_id, pad_id):
        from omegaconf import OmegaConf
        self.step, self.next_update = int(step), 0
        self.mask_id, self.pad_id = mask_id, pad_id
        self.assert_frozen()
        self.episode_count = len(episodes)
        self.zero_weight_episodes = sum(float(e.get('loss_weight', 1.)) <= 0 for e in episodes)
        rows = []
        for ep in episodes:
            rows.extend(TTR.build_action_rows(ep, reward=0., loss_weight=1.,
                max_segments=int(self.config.training.max_train_segments_per_episode), rng=self.rng))
        self.raw_segments = len(rows)
        build_cfg = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        # Anchor rows use separate optimizer windows.
        build_cfg.training.train_token_budget = 0
        build_cfg.training.align_fsdp_workloads = False
        self.built, _, _, _ = self.trainer._build_bucketed_action_rows(
            rows=rows, tokenizer=tokenizer, config=build_cfg, mask_id=mask_id, pad_id=pad_id,
            device=self.device, world_size=self.world, rank=self.rank,
            block_size=int(self.config.training.block_size), shrink=int(self.config.training.shrink),
            max_prompt_len=int(self.config.training.max_prompt_len),
            rollout_bucket_size=int(self.config.training.rollout_bucket_size),
            bucket_order=str(self.config.training.bucket_order), shuffle_seed=int(self.config.training.seed))
        if not self.built:
            raise RuntimeError('No replayable anchor states, including zero-reward groups')
        self.windows = make_anchor_windows(self.built, updates=self.updates,
            world_size=self.world, rank=self.rank, pad_id=pad_id,
            seed=int(self.config.training.seed) + int(self.cfg.seed_offset) + step,
            token_budget=int(self.cfg.token_budget), max_rows=int(self.cfg.batch_size_max),
            attention_budget=int(self.config.training.attention_element_budget))
        needed = sorted({i for window in self.windows for entry in window['batches']
                         if not entry['is_padding'] for i in entry['row_indices']})
        ref_batches = [dict(row_indices=ids, is_padding=False, pad_id=pad_id)
            for ids in pack_ids(needed, self.built, token_budget=int(self.cfg.token_budget),
                max_rows=int(self.cfg.ref_batch_size),
                attention_budget=int(self.config.training.attention_element_budget))]
        # Empty ranks must still enter the collective and every reference forward.
        ref_count = int(self._max_across_ranks(len(ref_batches)))
        while len(ref_batches) < ref_count:
            ref_batches.append(dict(row_indices=[self.rank % len(self.built)],
                                    is_padding=True, pad_id=pad_id))
        self.reference.eval()
        with torch.no_grad():
            for entry in ref_batches:
                batch = self._batch(entry)
                lp = self._logp(self.reference, batch).float().cpu()
                if not entry['is_padding']:
                    for i, row_id in enumerate(entry['row_indices']):
                        length = self.built[row_id]['L0'] + self.built[row_id]['L1']
                        self.built[row_id]['logp_sft'] = lp[i, :length].clone()
        if any('logp_sft' not in self.built[i] for i in needed):
            raise RuntimeError('Missing fixed SFT log-probabilities')
        self.probe_entry = dict(row_indices=[self.rank % len(self.built)], is_padding=False, pad_id=pad_id)
        self.reference_probe = self._probe(self.reference)
        if step == 1:
            delta = self._max_across_ranks((self._probe(learner) - self.reference_probe).abs().max())
            self.initial_reference_delta = delta
            if delta > float(self.cfg.initial_match_tolerance):
                raise RuntimeError(f'Learner and fixed SFT disagree at initialization: {delta}')
            if self.acc.is_main_process:
                print(f'[SFT anchor] initial reference/learner max logp difference={delta:.8g}', flush=True)
        self.sums = torch.zeros(4, device=self.device, dtype=torch.float64)
        if self.acc.is_main_process:
            print(f'[SFT anchor] step={step} episodes={len(episodes)} '
                  f'zero-weight={self.zero_weight_episodes} segments={len(rows)} '
                  f'2N rows={len(self.built)} reference cached', flush=True)
        self.acc.wait_for_everyone()

    def accumulate_update(self, learner, update):
        if update != self.next_update or update >= self.updates:
            raise RuntimeError(f'Anchor update order mismatch: {update}/{self.next_update}')
        learner.train()
        window = self.windows[update]
        for entry in window['batches']:
            batch = self._batch(entry, with_reference=True)
            logp_new = self._logp(learner, batch)
            weight = batch['loss_weight'].to(self.device)
            summed, token_stats = fixed_sft_kl_sum(logp_new, batch['logp_sft'].to(self.device),
                batch['pmask'].to(self.device), batch['loss_denominator'].to(self.device),
                weight, tail=float(self.cfg.log_ratio_tail))
            effective = summed * (self.coefficient * self.world / max(1, window['global_rows']))
            if not bool(torch.isfinite(effective)):
                raise FloatingPointError('Non-finite fixed SFT anchor loss')
            # The trainer also reduces each PPO microbatch: preserve sharded gradients.
            self.acc.backward(effective)
            self.sums += torch.stack((summed.detach(), weight.sum(), *token_stats)).double()
        self.next_update += 1

    def finish_updates(self, learner, optimizer, metrics):
        """Use the same update budget when PPO has fewer rows or no reward signal."""
        ppo_updates = int(metrics['action_updates'])
        while int(metrics['action_updates']) < self.updates:
            update = int(metrics['action_updates'])
            self.accumulate_update(learner, update)
            if self.config.training.max_grad_norm:
                self.acc.clip_grad_norm_(learner.parameters(), float(self.config.training.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad()
            metrics['action_updates'] += 1.
        if self.next_update != self.updates:
            raise RuntimeError('Anchor and optimizer update counts differ')
        self.assert_frozen()
        change = self._max_across_ranks((self._probe(self.reference) - self.reference_probe).abs().max())
        if change > float(self.cfg.reference_tolerance):
            raise RuntimeError(f'Fixed SFT reference probabilities changed: {change}')
        sums = self.sums.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums)
        kl_sum, count, tails, tokens = sums.cpu().tolist()
        if round(count) != len(self.built):
            raise RuntimeError(f'Anchor row coverage mismatch: {count}/{len(self.built)}')
        result = dict(sft_anchor_kl=kl_sum / max(1., count),
            sft_anchor_loss=self.coefficient * kl_sum / max(1., count),
            sft_anchor_rows=count, sft_anchor_tokens=tokens,
            sft_anchor_tail_fraction=tails / max(1., tokens),
            sft_anchor_zero_weight_episodes=float(self.zero_weight_episodes),
            sft_anchor_reference_change=change, sft_anchor_updates=float(self.next_update),
            sft_anchor_only_updates=float(self.updates-ppo_updates))
        metrics.update(result)
        if self.acc.is_main_process:
            path = Path(str(self.config.experiment.project))
            path.mkdir(parents=True, exist_ok=True)
            with (path/'sft_anchor_steps.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=self.step, **result), allow_nan=False) + '\n')
            print('[SFT anchor] ' + json.dumps(result, allow_nan=False), flush=True)
        self.built, self.windows = [], []
        return metrics
