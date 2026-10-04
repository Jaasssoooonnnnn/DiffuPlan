from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


def compute_diffuplan_edit_joint_marginal_loss(
    *,
    logits: torch.Tensor,
    edit_loss_mask: torch.Tensor,
    target_ids: torch.Tensor,
    target_lens: torch.Tensor,
    segment_kinds: torch.Tensor,
    segment_lens: torch.Tensor,
    record_types: torch.Tensor,
    expand_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return summed uniformly normalized marginal NLL and active-mask count.

    Each active mask slot may either emit the exact target token when it covers
    one token, or the expand token when it covers a target chunk of length >= 2.
    The partition boundaries between unresolved mask slots are latent, so this
    loss marginalizes all valid partitions with a uniform prior.
    """

    total_loss = torch.zeros((), dtype=torch.float32, device=logits.device)
    total_slots = torch.zeros((), dtype=torch.long, device=logits.device)

    if record_types is None:
        return total_loss, total_slots

    record_types = record_types.to(device=logits.device)
    edit_loss_mask = edit_loss_mask.to(device=logits.device, dtype=torch.bool)
    target_ids = target_ids.to(device=logits.device)
    target_lens = target_lens.to(device=logits.device)
    segment_kinds = segment_kinds.to(device=logits.device)
    segment_lens = segment_lens.to(device=logits.device)

    for batch_idx in range(logits.shape[0]):
        if int(record_types[batch_idx].item()) != 1:
            continue

        active_positions = torch.nonzero(edit_loss_mask[batch_idx], as_tuple=False).flatten()
        if active_positions.numel() == 0:
            continue

        target_len = int(target_lens[batch_idx].item())
        if target_len <= 0:
            raise ValueError("Diffuplan edit record has active masks but no target tokens.")
        if target_len < int(active_positions.numel()):
            raise ValueError(
                "Diffuplan edit record has more active masks than target tokens; "
                "delete-style edit states are not supported."
            )
        sample_target = target_ids[batch_idx, :target_len]
        sample_segments = _segment_groups(
            segment_kinds=segment_kinds[batch_idx],
            segment_lens=segment_lens[batch_idx],
            target_len=target_len,
        )
        if not sample_segments:
            sample_segments = [(int(active_positions.numel()), 0, target_len)]

        mask_cursor = 0
        active_log_probs = F.log_softmax(logits[batch_idx, active_positions].float(), dim=-1)
        for mask_count, target_start, target_end in sample_segments:
            if mask_count <= 0:
                continue
            group_log_probs = active_log_probs[mask_cursor : mask_cursor + mask_count]
            mask_cursor += mask_count
            if group_log_probs.shape[0] != mask_count:
                raise ValueError("Diffuplan edit segment metadata does not match active mask count.")

            group_target = sample_target[target_start:target_end]
            if group_target.numel() < mask_count:
                raise ValueError(
                    "Diffuplan edit segment has more masks than target tokens; "
                    "delete-style edit states are not supported."
                )

            group_loss = _joint_marginal_group_nll(
                log_probs=group_log_probs,
                target_ids=group_target,
                expand_token_id=expand_token_id,
            )
            total_loss = total_loss + group_loss
            total_slots = total_slots + int(mask_count)

        if mask_cursor != int(active_positions.numel()):
            raise ValueError("Diffuplan edit segment metadata left active masks unaccounted for.")

    return total_loss, total_slots


def _segment_groups(
    *,
    segment_kinds: torch.Tensor,
    segment_lens: torch.Tensor,
    target_len: int,
) -> List[Tuple[int, int, int]]:
    groups: List[Tuple[int, int, int]] = []
    cursor = 0
    group_start = 0
    group_masks = 0

    for kind, length in zip(segment_kinds.tolist(), segment_lens.tolist()):
        kind = int(kind)
        length = int(length)
        if kind < 0:
            break
        if length <= 0:
            continue
        start = cursor
        end = min(target_len, cursor + length)
        if kind == 1:
            group_masks += 1
        else:
            if group_masks:
                groups.append((group_masks, group_start, start))
                group_masks = 0
            group_start = end
        cursor = end

    if group_masks:
        groups.append((group_masks, group_start, target_len))
    return groups


def _joint_marginal_group_nll(
    *,
    log_probs: torch.Tensor,
    target_ids: torch.Tensor,
    expand_token_id: int,
) -> torch.Tensor:
    mask_count = int(log_probs.shape[0])
    target_len = int(target_ids.numel())
    neg_inf = torch.finfo(log_probs.dtype).min

    rows = [torch.cat((log_probs.new_zeros(1), log_probs.new_full((target_len,), neg_inf)))]

    for mask_idx in range(1, mask_count + 1):
        prev = rows[-1]
        prefix = torch.logcumsumexp(prev, dim=0)
        expand_log_prob = log_probs[mask_idx - 1, int(expand_token_id)]
        values = [log_probs.new_full((), neg_inf)]
        for target_pos in range(1, target_len + 1):
            token_id = int(target_ids[target_pos - 1].item())
            token_term = prev[target_pos - 1] + log_probs[mask_idx - 1, token_id]
            if target_pos >= 2:
                expand_term = prefix[target_pos - 2] + expand_log_prob
                value = torch.logsumexp(torch.stack((token_term, expand_term)), dim=0)
            else:
                value = token_term
            values.append(value)
        rows.append(torch.stack(values))

    # Positive m-part compositions of n have cardinality C(n - 1, m - 1).
    log_partition_count = (
        math.lgamma(target_len)
        - math.lgamma(mask_count)
        - math.lgamma(target_len - mask_count + 1)
    )
    return -rows[-1][target_len] + log_partition_count
