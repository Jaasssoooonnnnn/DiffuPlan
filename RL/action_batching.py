"""Pack 2N training rows into attention and optimizer windows."""
import torch


def pack_ids(ids, built, *, token_budget, max_rows, attention_budget=0):
    ordered = sorted(ids, key=lambda i: (built[i]['N'], built[i]['L0'], i))
    groups, group = [], []
    for row_id in ordered:
        rows = len(group) + 1
        length = built[row_id]['N']
        if group and (len(group) >= max_rows or rows * length > token_budget or
                      (attention_budget > 0 and rows * length * length > attention_budget)):
            groups.append(group)
            group = []
        group.append(row_id)
    if group:
        groups.append(group)
    return groups


def pack_training_windows(groups, built, plan, *, world_size, pad_id, token_budget, max_rows, attention_budget=0):
    result = []
    start = 0
    for end, window_rows in zip(plan['boundaries'], plan['window_rows']):
        # Original optimizer-window membership is fixed before repacking.
        ids = [i for group in groups[start * world_size:min(end * world_size, len(groups))] for i in group]
        packs = pack_ids(ids, built, token_budget=token_budget, max_rows=max_rows, attention_budget=attention_budget)
        entries = [dict(row_indices=ids, is_padding=False, pad_id=pad_id) for ids in packs]
        while len(entries) % world_size:
            entries.append(dict(row_indices=packs[0][:1], is_padding=True, pad_id=pad_id))
        for offset in range(0, len(entries), world_size):
            slot = entries[offset:offset + world_size]
            active_rows = sum(sum(float(built[i].get('loss_weight', 1.)) > 0 for i in e['row_indices'])
                              for e in slot if not e['is_padding'])
            active_ranks = sum(not e['is_padding'] for e in slot)
            for entry in slot:
                entry.update(slot_index=(len(result) + offset) // world_size,
                             slot_active_rows=active_rows, active_ranks=max(1, active_ranks),
                             original_update_end=offset + world_size == len(entries),
                             original_update_window_rows=window_rows)
        result.extend(entries)
        start = end
    return result


def collate_ragged(rows, entry, include_old_logp):
    ids = entry['row_indices']
    selected = [rows[i] for i in ids]
    n = max(r['N'] for r in selected)
    label_len = max(r['L0'] + r['L1'] for r in selected)
    b = len(selected)
    pad = int(entry.get('pad_id', 0))
    ext = torch.full((b, n), pad, dtype=torch.long)
    pos = torch.zeros_like(ext)
    labels = torch.full((b, label_len), pad, dtype=torch.long)
    pmask = torch.zeros((b, label_len), dtype=torch.bool)
    old = torch.zeros((b, label_len), dtype=torch.float32)
    for index, row in enumerate(selected):
        length = row['L0'] + row['L1']
        ext[index, :row['N']] = row['ext']
        pos[index, :row['N']] = row['tok_idx']
        labels[index, :length] = row['labels']
        pmask[index, :length] = row['pmask']
        if include_old_logp:
            if row.get('logp_old') is None and not entry['is_padding']:
                raise RuntimeError('Missing frozen old-policy logp')
            if row.get('logp_old') is not None:
                old[index, :length] = row['logp_old'][:length]
    is_padding = bool(entry.get('is_padding', False))
    out = dict(entry, ids=torch.tensor(ids), ext=ext, tok_idx=pos, labels=labels, pmask=pmask,
               reward=torch.tensor([0. if is_padding else r['reward'] for r in selected]),
               loss_weight=torch.tensor([0. if is_padding else r.get('loss_weight', 1.) for r in selected]),
               L0=torch.tensor([r['L0'] for r in selected]),
               L1=torch.tensor([r['L1'] for r in selected]),
               loss_denominator=torch.tensor([r.get('loss_denominator', r['L1']) for r in selected]), N=n)
    if include_old_logp:
        out['logp_old'] = old
    return out


def ragged_attention(l0, l1, n, block_size, dtype, device):
    """Same absolute block mask as scalar rows; only whole-row padding is added."""
    from action_geometry import action_attention
    return action_attention(l0, l1, n, block_size, dtype, device)
