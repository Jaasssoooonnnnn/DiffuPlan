"""TraceRL PPO losses for Action Mode.

Action rows use block-aligned 2N attention."""

from __future__ import annotations

import torch

EXPAND_TOKEN_ID = 156901


# =============================================================================
# ACTION 2N machinery -- TraceRL objective with absolute rollout block geometry
# =============================================================================

def collapse_k_unique(lst, k: int):
    if k <= 0:
        raise ValueError("k must be > 0")
    non_neg = [x for x in lst if x >= 0]
    if not non_neg:
        return lst
    uniq = sorted(set(non_neg))
    mapping = {}
    n = len(uniq)
    for idx, val in enumerate(uniq):
        group = idx // k
        end_idx = min((group + 1) * k - 1, n - 1)
        rep = uniq[end_idx]
        mapping[val] = rep
    return [mapping.get(x, x) for x in lst]


def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
    device = input_ids_b.device
    NB = (L1 + block_size - 1) // block_size

    step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
    step_pad[:L1] = step_map_b
    step_blk = step_pad.view(NB, block_size)

    valid = step_blk.ge(0)
    big = torch.iinfo(step_blk.dtype).max
    tmp = step_blk.masked_fill(~valid, big)
    min_vals, _ = tmp.min(dim=1, keepdim=True)

    pmask_blk = step_blk.eq(min_vals) & valid
    if not pmask_blk.any():
        return None, None, step_map_b, False

    ge_mask_blk = step_blk.ge(min_vals) & valid

    pmask_tail = pmask_blk.view(-1)[:L1]
    ge_mask_tail = ge_mask_blk.view(-1)[:L1]

    pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
    pmask_b[L0:] = pmask_tail

    tail = input_ids_b[L0:L0 + L1].clone()
    tail[ge_mask_tail] = mask_id

    extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
    extended_input_ids_b[:L0 + L1] = input_ids_b
    extended_input_ids_b[L0 + L1:] = tail

    new_step_map_b = step_map_b.clone()
    new_step_map_b[pmask_tail] = -1

    return extended_input_ids_b, pmask_b, new_step_map_b, True


def make_basic_block_attention(N: int, start_pos: int, block_size: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Absolute block attention; the shared prefix must end on a block boundary."""
    L0 = int(start_pos)
    L1 = (N - L0) // 2
    if L0 % block_size or L0 + 2 * L1 != N:
        raise ValueError("2N rows require a block-aligned prefix and length L0 + 2*L1")
    from action_geometry import action_attention
    return action_attention([L0], [L1], N, block_size, dtype, "cpu")


def process_pad(attn, input_ids, pad_id, start_pos):
    key_mask = (input_ids == pad_id).unsqueeze(1).unsqueeze(2)
    attn.masked_fill_(key_mask, float("-inf"))
    N = input_ids.shape[1]
    rng = torch.arange(N, device=attn.device)
    attn[:, :, rng, rng] = 0.0
    return attn


def _selected_token_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    selected_logits = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return selected_logits - torch.logsumexp(logits, dim=-1)


def collect_training_data(
    input_ids, step_map_list, reward, start_pos, block_size, mask_id, shrink,
    method="TraceRL", loss_weight=None,
):
    """Expand clean rows whose shared prefix ends on an absolute block boundary.

    Use collect_action_segment for raw prompt/response pairs. In these lower-level
    rows the tail can begin with known prompt tokens (step -1, never supervised).
    """
    if start_pos % block_size:
        raise ValueError("Use collect_action_segment to duplicate the partial prompt block")
    step_map_list = [list(steps) for steps in step_map_list]
    B, L = input_ids.shape
    L0 = start_pos
    L1 = L - L0
    if loss_weight is None:
        loss_weight = [1.0 for _ in range(B)]

    if method != "TraceRL":
        raise NotImplementedError(f"Method {method} not supported.")

    for b in range(B):
        step_map_i = step_map_list[b]
        for j in range(int((L1 - 1) / block_size) + 1):
            start = j * block_size
            end = min(L1, (j + 1) * block_size)
            step_map_list[b][start:end] = collapse_k_unique(step_map_i[start:end], shrink)

    step_map = torch.as_tensor(step_map_list, dtype=torch.long, device=input_ids.device)

    extended_input_ids_list, pmask_list, reward_list, loss_weight_list = [], [], [], []
    for b in range(B):
        step_b = step_map[b]
        while True:
            extended_b, pmask_b, step_b, has_any = one_round_vectorized(
                input_ids_b=input_ids[b], step_map_b=step_b, L0=L0, L1=L1,
                block_size=block_size, mask_id=mask_id,
            )
            if not has_any:
                break
            extended_input_ids_list.append(extended_b)
            pmask_list.append(pmask_b)
            reward_list.append(reward[b])
            loss_weight_list.append(float(loss_weight[b]))

    if not extended_input_ids_list:
        return None

    extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
    p_mask = torch.stack(pmask_list, dim=0).to(torch.bool)
    labels = extended_input_ids[:, :L].clone()

    pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(len(extended_input_ids), -1)
    pos_ids_resp = pos_ids[:, start_pos:]
    tok_idx_ext = torch.cat([pos_ids, pos_ids_resp], dim=1)

    return extended_input_ids, p_mask, tok_idx_ext, labels, reward_list, loss_weight_list


def collect_action_segment(prompt_ids, response_ids, step_map, *, reward, loss_weight,
                           block_size, mask_id, shrink, method="TraceRL",
                           conditioning_ids=None, conditioning_steps=None, action_loss_mask=None):
    """Build one trajectory before batching, preserving every real token position.

    Duplicate at most block_size-1 prompt tokens into the noisy tail. They stay
    known (step -1) and are never targets. A shared clean partial prompt block
    would otherwise either lose response context or expose clean response labels.
    """
    if len(response_ids) != len(step_map):
        raise ValueError("Response tokens and diffusion steps must be aligned")
    if conditioning_ids is not None:
        if conditioning_steps is None or action_loss_mask is None:
            raise ValueError("Complete action replay requires steps and a separate loss mask")
        if not (len(conditioning_ids) == len(conditioning_steps) == len(action_loss_mask)):
            raise ValueError("Complete action replay fields are not aligned")
        if list(conditioning_ids[:len(response_ids)]) != list(response_ids):
            raise ValueError("Visible action must be a prefix of its complete replay")
        response_ids, step_map = conditioning_ids, conditioning_steps
    if action_loss_mask is None:
        action_loss_mask = [True] * len(response_ids)
    supervised = [bool(keep) and int(step) >= 0 and int(tok) != mask_id
                  and int(tok) != EXPAND_TOKEN_ID
                  for keep, step, tok in zip(action_loss_mask, step_map, response_ids)]
    prompt_length = len(prompt_ids)
    l0 = prompt_length // block_size * block_size
    clean = torch.tensor([list(prompt_ids) + list(response_ids)], dtype=torch.long)
    tail_steps = [-1] * (prompt_length - l0) + list(step_map)
    result = collect_training_data(clean, [tail_steps], [reward], l0, block_size,
                                  mask_id, shrink, method=method, loss_weight=[loss_weight])
    if result is not None:
        ext, pmask, pos, labels, rewards, weights = result
        pmask &= torch.tensor([False] * prompt_length + supervised)[None]
        keep = pmask.any(dim=1)
        if not keep.any():
            result = None
        else:
            indices = keep.nonzero().flatten().tolist()
            result = (ext[keep], pmask[keep], pos[keep], labels[keep],
                      [rewards[i] for i in indices], [weights[i] for i in indices])
    return result, l0, clean.shape[1] - l0


def _action_token_logp(model, extended_input_ids, attention_mask, tok_idx_ext,
                       labels, p_mask, L0, L1, temperature=1.0, banned_token_ids=()):
    from action_policy import token_logp
    ragged = torch.is_tensor(L0)
    sparse = getattr(model, "_rl_sparse_logp", False) and p_mask is not None
    if not sparse and not ragged:
        logits = model(input_ids=extended_input_ids, attention_mask=attention_mask,
                       position_ids=tok_idx_ext).logits
        logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1:, :]], dim=1)
        return token_logp(logits, labels, temperature=temperature, banned_token_ids=banned_token_ids)
    selected = p_mask.nonzero(as_tuple=False)
    row, col = selected[:, 0], selected[:, 1]
    if ragged:
        prefix = L0.to(col.device)[row]
        response = L1.to(col.device)[row]
    else:
        prefix, response = L0, L1
    ext_col = torch.where(col < prefix, col, col + response)
    indices = row * extended_input_ids.shape[1] + ext_col
    if sparse:
        output = model(input_ids=extended_input_ids, attention_mask=attention_mask,
                       position_ids=tok_idx_ext, rl_token_indices=indices,
                       rl_token_labels=labels[row, col], rl_sampling_temperature=temperature,
                       rl_banned_token_ids=banned_token_ids)
        logp = output["selected_logp"]
    else:
        logits = model(input_ids=extended_input_ids, attention_mask=attention_mask,
                       position_ids=tok_idx_ext).logits
        logp = token_logp(logits[row, ext_col], labels[row, col],
                         temperature=temperature, banned_token_ids=banned_token_ids)
    return logp.new_zeros(labels.shape).index_put((row, col), logp)


def _action_attention(basic, inputs, pad_id, start_pos, L0, L1, block_size, dtype):
    if basic is None:
        from action_batching import ragged_attention
        attention = ragged_attention(L0, L1, inputs.shape[1], block_size, dtype, inputs.device)
    else:
        attention = basic.repeat(inputs.shape[0], 1, 1, 1)
    # Geometry already excludes right padding. EOS can be a REAL context token
    # (and is also the pad ID); masking by token value would corrupt replay.
    lengths = torch.as_tensor(L0, device=inputs.device) + 2 * torch.as_tensor(L1, device=inputs.device)
    lengths = lengths.reshape(-1, 1)
    idx = torch.arange(inputs.shape[1], device=inputs.device)
    padded_query = idx[None] >= lengths
    attention[:, 0, idx, idx] = torch.where(padded_query, 0.0, attention[:, 0, idx, idx])
    return attention


def action_ppo_loss(
    *,
    model,
    extended_input_ids,
    p_mask,
    tok_idx_ext,
    labels,
    adv,
    loss_weight,
    logp_old_tok,
    basic_block_attention,
    pad_id,
    start_pos,
    L0,
    L1,
    eps: float,
    beta: float,
    use_kl_estimator_k3: bool,
    block_size: int = 32,
    reuse_current_as_old: bool = False,
    loss_denominator=None,
    temperature=1.0,
    banned_token_ids=(),
):
    """PPO-clip on block-diffusion rows, independently normalized from their layout."""
    adv = adv.detach()
    loss_weight = loss_weight.detach().to(dtype=adv.dtype)
    B = extended_input_ids.shape[0]
    active_sample_denom = loss_weight.sum().clamp_min(1.0)
    attention_mask = _action_attention(basic_block_attention, extended_input_ids,
        pad_id, start_pos, L0, L1, block_size, getattr(model, "_rl_attention_dtype", getattr(model, "dtype", torch.bfloat16)))

    logp_new_tok = _action_token_logp(model, extended_input_ids, attention_mask,
                                     tok_idx_ext, labels, p_mask, L0, L1,
                                     temperature, banned_token_ids)

    if reuse_current_as_old:
        # Valid only before the outer step's first optimizer update, with a
        # deterministic decoder: current weights still equal the frozen policy.
        logp_old_tok = logp_new_tok.detach()
    ratio = logp_new_tok - logp_old_tok
    ratio = torch.where(p_mask, ratio, torch.zeros_like(ratio)).clamp(-10.0, 10.0)
    ratio = torch.exp(ratio)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)

    adv_tok = adv.unsqueeze(1)
    surrogate_tok = torch.min(ratio * adv_tok, clipped * adv_tok)
    surrogate_tok = surrogate_tok * p_mask
    # Geometry includes the partial prompt block; keep loss normalization separate.
    denominator = L1 if loss_denominator is None else loss_denominator
    denominator = (denominator.to(adv.device).clamp_min(1) if torch.is_tensor(denominator)
                   else max(1, denominator))
    surrogate_tok = surrogate_tok.sum(dim=1) / denominator
    surrogate_tok = surrogate_tok * loss_weight
    policy_loss = -(surrogate_tok.sum() / active_sample_denom)

    kl_loss = torch.tensor(0.0, device=policy_loss.device)
    if beta > 0:
        kl_seq = logp_new_tok - logp_old_tok
        kl_seq = torch.where(p_mask, kl_seq, torch.zeros_like(kl_seq))
        if use_kl_estimator_k3:
            t = (-kl_seq).clamp(-10.0, 10.0)
            kl_seq = t.exp() - 1.0 + kl_seq
        kl_seq = (kl_seq * p_mask).sum(dim=1) / denominator
        kl_seq = kl_seq * loss_weight
        kl_loss = beta * kl_seq.sum() / active_sample_denom

    total = policy_loss + kl_loss
    return total, policy_loss.detach(), kl_loss.detach()


@torch.no_grad()
def compute_action_logp_old_tok(
    *, model, extended_input_ids, tok_idx_ext, labels, basic_block_attention, pad_id, start_pos, L0, L1, p_mask=None, block_size=32,
    temperature=1.0, banned_token_ids=(),
):
    """Reference forward for action rows: per-token logprob at label positions."""
    B = extended_input_ids.shape[0]
    attention_mask = _action_attention(basic_block_attention, extended_input_ids,
        pad_id, start_pos, L0, L1, block_size, getattr(model, "_rl_attention_dtype", getattr(model, "dtype", torch.bfloat16)))
    return _action_token_logp(model, extended_input_ids, attention_mask,
                              tok_idx_ext, labels, p_mask, L0, L1,
                              temperature, banned_token_ids).float().detach()
