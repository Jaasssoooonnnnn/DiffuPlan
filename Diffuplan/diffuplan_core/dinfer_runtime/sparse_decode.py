"""Sparse action vocabulary projection with dense-position random-noise indexing.

Sampling preserves noise coordinates and RNG advancement across decoder calls."""

from dataclasses import dataclass
from types import MethodType, SimpleNamespace
import torch
import torch.nn.functional as F


@dataclass
class SparseLogits:
    values: object
    indices: object
    batch: int
    width: int

    def __getitem__(self, key):
        rows, columns = key
        if rows != slice(None) or not isinstance(columns, slice):
            raise ValueError("Unsupported sparse-logit slice")
        start, stop, stride = columns.indices(self.width)
        if stride != 1:
            raise ValueError("Sparse logit slice must be contiguous")
        col = self.indices % self.width
        keep = (col >= start) & (col < stop)
        indices = (self.indices // self.width) * (stop - start) + (col - start)
        return SparseLogits(
            None if self.values is None else self.values[keep],
            indices[keep],
            self.batch,
            stop - start,
        )


def prepare_head(engine):
    head = engine.model.lm_head
    if not getattr(head, "_sparse_decode_installed", False):
        original = head.forward
        engine._dense_lm_head_forward = original

        def forward(self, hidden, *args, **kwargs):
            if getattr(engine, "_decode_hidden_only", False):
                return hidden
            return original(hidden, *args, **kwargs)

        head.forward = MethodType(forward, head)
        head._sparse_decode_installed = True
    engine._decode_hidden_only = True


def finish_install(engine):
    engine._decode_hidden_only = False
    original = type(engine.model_runner).forward.__get__(
        engine.model_runner, type(engine.model_runner)
    )

    @torch.no_grad()
    def forward(*args, **kwargs):
        ids = args[0] if args else kwargs["input_ids"]
        original_batch = ids.shape[0]
        engine._decode_hidden_only = True
        try:
            output = original(*args, **kwargs)
        finally:
            engine._decode_hidden_only = False
        indices = (ids == engine.decoder.mask_id).flatten().nonzero().flatten()
        hidden = output.logits[: ids.shape[0]]
        values = (
            engine._dense_lm_head_forward(
                hidden.reshape(-1, hidden.shape[-1]).index_select(0, indices)
            )
            if indices.numel()
            else None
        )
        return SimpleNamespace(
            logits=SparseLogits(values, indices, original_batch, ids.shape[1]),
            past_key_values=output.past_key_values,
            hidden_states=hidden,
        )

    engine.model_runner.forward = forward
    original_decode = type(engine.decoder).decode.__get__(
        engine.decoder, type(engine.decoder)
    )

    @torch.no_grad()
    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        if not isinstance(logits, SparseLogits):
            before = x[:, block_start:block_end] == self.mask_id
            original_decode(
                logits, block_start, block_end, x, iter_threshold=iter_threshold
            )
            return before & (x[:, block_start:block_end] != self.mask_id)
        if logits.values is None:
            if self.temperature > 0:
                torch.randint(
                    0,
                    torch.iinfo(torch.int64).max,
                    (1,),
                    device=logits.indices.device,
                    dtype=torch.int64,
                )
            return None
        assert logits.width == block_end - block_start
        z = self._apply_banned_tokens(logits.values)
        if self._use_deterministic_zero_temp():
            # Preserve the original eager zero-temperature reduction path.
            choices = z.argmax(-1)
            prob = F.softmax(
                z.to(torch.float64 if self.use_float64 else torch.float32), dim=-1
            )
            confidence = prob.gather(-1, choices[:, None]).flatten()
        else:
            from .indexed_sampling import sample

            choices, confidence = sample(
                z,
                logits.indices,
                logits.batch,
                logits.width,
                float(self.temperature),
                bool(self.use_float64),
            )
        from .indexed_sampling import apply_choices

        threshold = float(self.threshold if iter_threshold is None else iter_threshold)
        updated, transfer = apply_choices(
            x[:, block_start:block_end],
            logits.indices,
            choices,
            confidence,
            self.mask_id,
            threshold,
            self._use_deterministic_zero_temp(),
        )
        x[:, block_start:block_end] = updated
        return transfer

    engine.decoder.decode = MethodType(decode, engine.decoder)
    engine.decoder._returns_transfer_mask = True
