"""Model/token helpers for reserved expand-token wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class ExpandTokenInfo:
    token_id: int
    token_text: str
    alias: str = "<expand>"
    initialized: bool = False


def resolve_expand_token(tokenizer: Any, preferred_text: str = "<|reserved_token_6|>") -> ExpandTokenInfo:
    token_id = tokenizer.convert_tokens_to_ids(preferred_text)
    if token_id is None or int(token_id) < 0:
        raise ValueError(f"Could not resolve reserved expand token: {preferred_text}")
    return ExpandTokenInfo(token_id=int(token_id), token_text=preferred_text)


def initialize_expand_token_mean(model: Any, token_id: int) -> bool:
    """Initialize a reserved token row to the mean embedding/lm-head vector."""

    if torch is None:
        raise RuntimeError("torch is required to initialize model weights.")

    changed = False
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None and token_id < input_embeddings.weight.shape[0]:
            weight = input_embeddings.weight
            weight[token_id].copy_(weight.mean(dim=0))
            changed = True

        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and token_id < output_embeddings.weight.shape[0]:
            weight = output_embeddings.weight
            weight[token_id].copy_(weight.mean(dim=0))
            changed = True

    return changed
