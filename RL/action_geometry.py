"""Block-diffusion attention for block-aligned 2N trajectory rows.

Layout: [clean prefix | clean tail | noisy tail]. A partial prompt block belongs
to both tail copies; known prompt tokens provide conditioning context."""
import torch


def action_attention(l0, l1, n, block_size, dtype, device):
    l0 = torch.as_tensor(l0, device=device).reshape(-1, 1, 1)
    l1 = torch.as_tensor(l1, device=device).reshape(-1, 1, 1)
    q = torch.arange(n, device=device)[None, :, None]
    k = torch.arange(n, device=device)[None, None, :]
    clean_length = l0 + l1
    # All clean states, including the prompt, use absolute block-causal attention.
    clean_allowed = ((q < clean_length) & (k < clean_length)
                     & (k // block_size <= q // block_size))
    noisy_position = q - l1
    block_start = (noisy_position // block_size) * block_size
    noisy_allowed = ((q >= clean_length) & (q < l0 + 2 * l1)
                     & ((k < block_start)
                        | ((k >= block_start + l1)
                           & (k < block_start + l1 + block_size)
                           & (k >= clean_length))))
    allowed = (clean_allowed | noisy_allowed) & (k < l0 + 2 * l1)
    return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill_(
        ~allowed, float("-inf"))[:, None]
