"""Map physical prefix-cache storage to real block-causal token positions."""
from contextlib import contextmanager
import torch


def cache_attention_mask(position_ids, capacity, block_length):
    """dInfer writes this call's K/V into the physical cache TAIL.

    Only [0, query_start) contains reusable prefix K/V. The gap before the tail
    contains reserve/stale slots, including old copies of a recomputed block.
    Values in that gap must never participate in the attention normalization.
    """
    batch, width = position_ids.shape
    if capacity < width:
        raise ValueError("KV cache is shorter than its current query")
    physical = torch.arange(capacity, device=position_ids.device)[None, :]
    start = position_ids[:, :1]
    tail = physical >= capacity - width
    key_position = torch.where(tail, start + physical - (capacity - width), physical)
    valid = (physical < start) | tail
    return valid[:, None, :] & (key_position[:, None, :] // block_length <=
                                position_ids[:, :, None] // block_length)


def install_cache_conditioning(engine):
    original = engine.model_runner.forward

    @torch.no_grad()
    def forward(input_ids=None, position_ids=None, inputs_embeds=None,
                pp_proxy_tensors=None, past_key_values=None, replace_position=None,
                use_cache=False, attention_mask=None):
        if past_key_values is not None:
            data = getattr(past_key_values, "_data", past_key_values)
            if position_ids is None:
                raise ValueError("Cached action forward requires absolute positions")
            attention_mask = cache_attention_mask(position_ids, data.shape[4], engine.block_length)
        return original(input_ids=input_ids, position_ids=position_ids,
                        inputs_embeds=inputs_embeds, pp_proxy_tensors=pp_proxy_tensors,
                        past_key_values=past_key_values, replace_position=replace_position,
                        use_cache=use_cache, attention_mask=attention_mask)

    engine.model_runner.forward = forward


@contextmanager
def masked_decode_graphs():
    """The upstream single-block graph omits masks; capture it WITH a mask input."""
    from dinfer.decoding.diffusion_runner import CudaGraphRunner
    original = CudaGraphRunner.capture_one_batch_size

    def capture(self, bs, forward, is_decode_phase=True, length=0, cache_length=0, use_mask=False):
        return original(self, bs, forward, is_decode_phase, length, cache_length,
                        use_mask=bool(use_mask or is_decode_phase))

    CudaGraphRunner.capture_one_batch_size = capture
    try:
        yield
    finally:
        CudaGraphRunner.capture_one_batch_size = original
