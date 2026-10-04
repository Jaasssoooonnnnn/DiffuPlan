"""Action-token probabilities with the sampler temperature and vocabulary support.

Position selection uses decoder confidence scores."""
import math
import torch


def validate_temperature(temperature):
    value = float(temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Action PPO requires a finite positive rollout temperature")
    return value


def token_logp(logits, labels, *, temperature=1.0, banned_token_ids=()):
    temperature = validate_temperature(temperature)
    work = logits if logits.dtype == torch.float64 else logits.float()
    work = work / temperature
    banned = sorted({int(t) for t in banned_token_ids if 0 <= int(t) < work.shape[-1]})
    if banned:
        indices = torch.tensor(banned, device=work.device, dtype=torch.long)
        work = work.index_fill(-1, indices, float("-inf"))
    return work.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - work.logsumexp(-1)
