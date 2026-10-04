import inspect
import logging
from typing import List, Optional, Sequence

import torch


logger = logging.getLogger(__name__)


def _normalize_stop_sequences(
    stop_token_sequences: Optional[Sequence[Sequence[int]]],
) -> List[List[int]]:
    if not stop_token_sequences:
        return []
    normalized: List[List[int]] = []
    for seq in stop_token_sequences:
        if not seq:
            continue
        normalized.append([int(t) for t in seq])
    return normalized


def patch_block_diffusion_llm_generate() -> bool:
    """Patch dinfer BlockDiffusionLLM with incremental stop + per-seq budget control."""
    try:
        from dinfer import BlockDiffusionLLM
        from dinfer.decoding.utils import TokenArray
    except Exception as e:
        logger.warning("skip dinfer patch: import failed: %s", e)
        return False

    if getattr(BlockDiffusionLLM, "_trace_rl_generate_patch_applied", False):
        return True

    required = {
        "return_step_map",
        "stop_token_sequences",
        "stop_check_stride_blocks",
        "max_new_tokens_per_seq",
    }
    current_sig = inspect.signature(BlockDiffusionLLM.naive_batching_generate)
    if required.issubset(set(current_sig.parameters.keys())):
        BlockDiffusionLLM._trace_rl_generate_patch_applied = True
        logger.debug("dinfer patch skipped: BlockDiffusionLLM already supports required args")
        return True

    original_init = BlockDiffusionLLM.__init__
    original_naive = BlockDiffusionLLM.naive_batching_generate
    original_dynamic = BlockDiffusionLLM.dynamic_batching_generate

    @torch.no_grad()
    def patched_naive_batching_generate(
        self,
        prompt,
        gen_length=128,
        block_length=128,
        return_step_map=False,
        stop_token_sequences=None,
        stop_check_stride_blocks=1,
        max_new_tokens_per_seq=None,
    ):
        batch_size = int(prompt.shape[0])
        prompt_length = int(prompt.shape[1])
        num_blocks = (prompt_length + int(gen_length) + int(block_length) - 1) // int(block_length)
        total_length = num_blocks * int(block_length)
        new_gen_length = total_length - prompt_length

        mask_length = (
            total_length
            + int(block_length)
            - 1
        ) // int(block_length) * int(block_length)
        attn_mask_num_blocks = mask_length // int(block_length)

        block_mask = torch.tril(
            torch.ones(
                attn_mask_num_blocks,
                attn_mask_num_blocks,
                device=self.model.device,
                dtype=torch.bool,
            )
        )
        bd_attn_mask = (
            block_mask.repeat_interleave(int(block_length), dim=0)
            .repeat_interleave(int(block_length), dim=1)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        pos_ids = torch.arange(total_length, device=self.model.device).unsqueeze(0).expand(batch_size, -1)

        x = TokenArray(prompt, new_gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, int(block_length))
        first_block_start = int(it._get_first_block_start())
        kv_cache = self.cache_factory.create()

        prefill_blocks = first_block_start // int(block_length)
        prefill_length = max(prefill_blocks * int(block_length), int(block_length))
        self.block_runner.prefill(
            self.model,
            x[:, :prefill_length],
            kv_cache,
            pos_ids[:, :prefill_length],
            bd_attn_mask[:, :prefill_length, :prefill_length],
            self.prefilling_limit,
            int(block_length),
        )

        self.diff_iteration.iter_no = 0

        stop_sequences = _normalize_stop_sequences(stop_token_sequences)
        tail_keep = max((len(seq) for seq in stop_sequences), default=1) - 1
        tails = [[] for _ in range(batch_size)]
        stop_stride = max(1, int(stop_check_stride_blocks or 1))
        # These values are static through decoding. Transfer once, not repeatedly
        # through .item() for each sequence, stop tag and diffusion block.
        prompt_lens_cpu = (prompt != self.decoder.mask_id).sum(dim=1).tolist()
        counts = [0] * batch_size
        finished = [False] * batch_size
        cutoffs = [total_length] * batch_size
        force_finished = torch.zeros(batch_size, dtype=torch.bool, device=prompt.device)
        if max_new_tokens_per_seq is None:
            budgets_cpu = [int(new_gen_length)] * batch_size
        else:
            budgets_cpu = torch.as_tensor(max_new_tokens_per_seq).flatten().tolist()
            if len(budgets_cpu) == 1:
                budgets_cpu *= batch_size
            if len(budgets_cpu) != batch_size:
                raise ValueError("max_new_tokens_per_seq batch size mismatch")
            budgets_cpu = [max(0, int(value)) for value in budgets_cpu]
        all_positions = torch.arange(total_length, device=prompt.device)[None, :]

        # Track per-token diffusion step (mask -> non-mask transition).
        # This preserves the training signal used by TraceRL; block_id-only maps
        # collapse too aggressively and degrade rollout traces.
        step_map = torch.full(
            (batch_size, total_length),
            -1,
            dtype=torch.long,
            device=prompt.device,
        )
        global_step = 0
        # Keep sampled terminal-block tokens BEFORE stop/budget handling overwrites
        # them with EOS. They are conditioning state, not necessarily loss targets.
        replay_tokens = x.data.clone() if return_step_map else None
        replay_ends = list(prompt_lens_cpu)
        self._last_action_replay = None

        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            if self.backend == "vllm":
                cross_block_attn_mask = bd_attn_mask[
                    :, block_loc.start - int(block_length) : block_loc.end, : block_loc.end
                ]
            else:
                cross_block_attn_mask = torch.ones(
                    batch_size,
                    2 * int(block_length),
                    kv_cache.past_key_values._data.shape[4],
                    device=prompt.device,
                    dtype=torch.bool,
                )
                cross_block_attn_mask[:, : int(block_length), -int(block_length) :].fill_(False)

            original_decode = self.decoder.decode

            def _decode_with_step_map(logits, block_start, block_end, token_array, *args, **kwargs):
                nonlocal global_step
                token_data = token_array.data if hasattr(token_array, "data") else token_array
                if token_data.shape[0] != batch_size:
                    return original_decode(logits, block_start, block_end, token_array, *args, **kwargs)

                if getattr(self.decoder, "_returns_transfer_mask", False):
                    global_step += 1
                    transfer = original_decode(logits, block_start, block_end, token_array, *args, **kwargs)
                    if transfer is not None:
                        step_map[:, block_start:block_end].masked_fill_(transfer, global_step)
                    return transfer

                before_mask = token_data[:, block_start:block_end] == self.decoder.mask_id
                global_step += 1
                out = original_decode(logits, block_start, block_end, token_array, *args, **kwargs)
                after_mask = token_data[:, block_start:block_end] == self.decoder.mask_id
                changed = before_mask & (~after_mask)
                block_step = step_map[:, block_start:block_end]
                block_step.masked_fill_(changed, global_step)
                return out

            self.decoder.decode = _decode_with_step_map
            try:
                decode_compl = self.block_runner.decode(
                    self.model,
                    self.decoder,
                    x,
                    kv_cache,
                    block,
                    block_loc,
                    block_id,
                    pos_ids,
                    bd_attn_mask,
                    int(block_length),
                    cross_block_attn_mask,
                )
            finally:
                self.decoder.decode = original_decode

            if replay_tokens is not None:
                for seq_id in range(batch_size):
                    if not finished[seq_id]:
                        replay_tokens[seq_id, block_loc.start:block_loc.end].copy_(
                            x.data[seq_id, block_loc.start:block_loc.end])
                        replay_ends[seq_id] = int(block_loc.end)

            if (block_id + 1) % stop_stride == 0:
                # A single small D2H transfer replaces dozens of synchronizing
                # nonzero/item calls per tag and sequence in the original loop.
                block_cpu = x.data[:, block_loc.start:block_loc.end].tolist()
                changed_finished = False
                for seq_id in range(batch_size):
                    if finished[seq_id]:
                        continue
                    prompt_len = int(prompt_lens_cpu[seq_id])
                    seq_start = max(int(block_loc.start), prompt_len)
                    budget = budgets_cpu[seq_id]
                    if seq_start >= int(block_loc.end):
                        if counts[seq_id] >= budget:
                            cutoffs[seq_id] = prompt_len + budget
                            finished[seq_id] = True
                            changed_finished = True
                        continue
                    tokens = block_cpu[seq_id][seq_start - int(block_loc.start):]
                    seg_len = next((i for i, token in enumerate(tokens)
                                    if token in (self.decoder.mask_id, self.decoder.eos_id)), len(tokens))
                    if seg_len <= 0 and counts[seq_id] >= budget:
                        cutoffs[seq_id] = prompt_len + budget
                        finished[seq_id] = True
                        changed_finished = True
                        continue
                    keep = min(seg_len, max(0, budget - counts[seq_id]))
                    truncate = keep < seg_len
                    if keep > 0 and not truncate and stop_sequences:
                        combined = tails[seq_id] + tokens[:keep]
                        ends = []
                        for sequence in stop_sequences:
                            size = len(sequence)
                            for pos in range(len(combined) - size + 1):
                                if combined[pos:pos + size] == sequence:
                                    ends.append(pos + size)
                                    break
                        if ends:
                            keep = max(0, min(ends) - len(tails[seq_id]))
                            truncate = True
                    if keep > 0:
                        counts[seq_id] += keep
                        tails[seq_id] = ((tails[seq_id] + tokens[:keep])[-tail_keep:]
                                         if tail_keep > 0 else [])
                    if truncate or counts[seq_id] >= budget:
                        cutoffs[seq_id] = seq_start + keep
                        finished[seq_id] = True
                        changed_finished = True
                if changed_finished:
                    cutoff_tensor = torch.tensor(cutoffs, device=prompt.device)[:, None]
                    x.data.masked_fill_(all_positions >= cutoff_tensor, self.decoder.eos_id)
                    force_finished = torch.tensor(finished, dtype=torch.bool, device=prompt.device)

            finished = [old or bool(done) for old, done in zip(finished, decode_compl.tolist())]
            decode_compl = decode_compl | force_finished
            if torch.all(decode_compl) and self.early_stop:
                break

        logger.debug("The number of diffusion iterations: %s", self.num_forwards)
        # TokenArray.get_generated_tokens removes ALL EOS positions for batch=1,
        # shifting token coordinates relative to step_map. Keep the canvas here;
        # the engine slices the generated region before trimming its first EOS.
        out_tokens = x.data.clone()
        out_tokens.masked_fill_(out_tokens == self.decoder.mask_id, self.decoder.eos_id)
        if return_step_map:
            self._last_action_replay = {
                "token_ids": replay_tokens.cpu(), "step_map": step_map.cpu(),
                "end_positions": replay_ends,
            }
            return out_tokens, step_map
        return out_tokens

    @torch.no_grad()
    def patched_dynamic_batching_generate(
        self,
        prompt,
        gen_length=128,
        block_length=128,
        return_step_map=False,
        stop_token_sequences=None,
        stop_check_stride_blocks=1,
        max_new_tokens_per_seq=None,
    ):
        # Dynamic path in upstream dinfer does not expose step-map or stop hooks.
        # Use patched naive path when these hooks are requested.
        if (
            return_step_map
            or stop_token_sequences is not None
            or max_new_tokens_per_seq is not None
            or int(stop_check_stride_blocks or 1) != 1
        ):
            return patched_naive_batching_generate(
                self,
                prompt=prompt,
                gen_length=gen_length,
                block_length=block_length,
                return_step_map=return_step_map,
                stop_token_sequences=stop_token_sequences,
                stop_check_stride_blocks=stop_check_stride_blocks,
                max_new_tokens_per_seq=max_new_tokens_per_seq,
            )
        return original_dynamic(self, prompt=prompt, gen_length=gen_length, block_length=block_length)

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.use_naive_batching or self.backend != "sglang":
            self.generate = self.naive_batching_generate
        else:
            self.generate = self.dynamic_batching_generate

    BlockDiffusionLLM.__init__ = patched_init
    BlockDiffusionLLM.naive_batching_generate = patched_naive_batching_generate
    BlockDiffusionLLM.dynamic_batching_generate = patched_dynamic_batching_generate

    BlockDiffusionLLM._trace_rl_original_init = original_init
    BlockDiffusionLLM._trace_rl_original_naive_batching_generate = original_naive
    BlockDiffusionLLM._trace_rl_original_dynamic_batching_generate = original_dynamic
    BlockDiffusionLLM._trace_rl_generate_patch_applied = True
    logger.debug("applied TraceRL dinfer BlockDiffusionLLM patch")
    return True
