"""Preserve dInfer's compiled Gumbel noise indexing and transfer decisions."""

import torch
import torch.nn.functional as F


@torch.compile(dynamic=True)
def sample(logits, indices, batch, width, temperature, use_float64):
    if temperature == 0:
        choices = logits.argmax(-1)
    else:
        noise = torch.rand(
            (batch, width, logits.shape[-1]), dtype=torch.float64, device=logits.device
        )
        noise = noise.reshape(-1, logits.shape[-1]).index_select(0, indices)
        perturbed = logits.to(torch.float64).exp() / (
            (-torch.log(noise)) ** temperature
        )
        choices = perturbed.argmax(-1)
    probs = F.softmax(
        logits.to(torch.float64 if use_float64 else torch.float32), dim=-1
    )
    return choices, probs.gather(-1, choices[:, None]).flatten()


@torch.compile(dynamic=True)
def apply_choices(
    current, indices, choices, confidence, mask_id, threshold, deterministic
):
    picked = current.flatten().clone()
    picked[indices] = choices
    picked = picked.reshape_as(current)
    conf = torch.full(
        current.shape, float("-inf"), device=current.device, dtype=confidence.dtype
    )
    conf.view(-1)[indices] = torch.where(choices != mask_id, confidence, float("-inf"))
    maximum, which = conf.max(-1)
    if deterministic:
        high = conf >= threshold
        low = torch.zeros_like(high).scatter_(1, which[:, None], True)
        transfer = torch.where((maximum >= threshold)[:, None], high, low) & (
            conf > float("-inf")
        )
    else:
        actual = (maximum - 1e-5).clamp(-1000, threshold)[:, None]
        transfer = conf >= actual
    transfer = transfer & (current == mask_id)
    return torch.where(transfer, picked, current), transfer
