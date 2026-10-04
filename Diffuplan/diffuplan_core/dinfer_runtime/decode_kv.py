"""Avoid redundant decode KV transfers without changing cache contents."""

import bisect
import weakref
from types import MethodType
import torch


def install_replay_cache(engine):
    runner = engine.model_runner.graph_runner
    if runner is None:
        return

    def prepare(
        self,
        input_ids,
        position_ids,
        past_key_values,
        is_decode_phase,
        length,
        attention_mask,
        cache_length,
    ):
        raw_bs = input_ids.shape[0]
        raw_num_token = raw_bs * length
        bs = self.capture_bs[bisect.bisect_left(self.capture_bs, raw_bs)]
        self.input_ids[:raw_num_token].copy_(input_ids.flatten())
        self.position_ids[:raw_num_token].copy_(position_ids.flatten())
        if is_decode_phase:
            if hasattr(self, "_bounded_kv_views"):
                self.past_key_values = self._bounded_kv_views[cache_length]
            minimal_length = min(
                self.past_key_values.shape[4], past_key_values.shape[4], cache_length
            )
            if minimal_length != cache_length:
                raise ValueError("Graph KV capacity is too small")
            ref, version, old_bs, old_length = getattr(
                self, "_last_kv_input", (lambda: None, -1, -1, -1)
            )
            # Identity protects against recycled storage pointers between requests.
            try:
                current_version = past_key_values._version
            except RuntimeError:  # Inference-mode tensors have no version counter.
                current_version = None
            if (
                current_version is None
                or ref() is not past_key_values
                or version != current_version
                or old_bs != raw_bs
                or old_length != minimal_length
            ):
                self.past_key_values[:, :, :raw_bs, :, :minimal_length].copy_(
                    past_key_values[:, :, :raw_bs, :, :minimal_length]
                )
                self._last_kv_input = (
                    weakref.ref(past_key_values),
                    current_version,
                    raw_bs,
                    minimal_length,
                )
            # Positions >= cache_length are outside the captured graph's view.
        if not is_decode_phase and hasattr(self, "_last_kv_input"):
            del self._last_kv_input
        if attention_mask is not None:
            limit = min(self.attention_mask.shape[2], attention_mask.shape[2])
            if is_decode_phase and limit < cache_length:
                self.attention_mask[:raw_bs, :length, limit:cache_length].fill_(False)
            self.attention_mask[:raw_bs, :length, :limit].copy_(
                attention_mask[:, :, :limit]
            )
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs
        self.is_decode_phase = is_decode_phase
        self.length = length

    runner.replay_prepare = MethodType(prepare, runner)


def install_block_updates(engine):
    factory = engine.dllm.cache_factory
    original_create = factory.create

    def create(*args, **kwargs):
        manager = original_create(*args, **kwargs)
        original = manager.range_update

        def update(self, past_key_values, range_start, range_end, block_length):
            if block_length <= 0 or self.past_key_values is None or range_start != 0:
                return original(past_key_values, range_start, range_end, block_length)
            current = self.past_key_values
            # The original constructor reserves a 64-token lookahead after each update.
            self.extend_cache(range_end + 64)
            new = torch.stack([v[:, :, -block_length:] for v in past_key_values], dim=0)
            new = new.reshape(current._data.shape[0], 2, *new.shape[1:])
            current._data[:, :, :, :, range_end - block_length : range_end].copy_(new)

        manager.range_update = MethodType(update, manager)
        return manager

    factory.create = create


@torch.library.custom_op(
    "diffuplan::write_kv_tail", mutates_args=("cache_k", "cache_v")
)
def _write_kv_tail(
    cache_k: torch.Tensor, cache_v: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> None:
    # An opaque mutation prevents Inductor functionalization from turning a
    # tail write back into a full-cache slice_scatter allocation.
    cache_k[:, :, -key.shape[2] :].copy_(key)
    cache_v[:, :, -value.shape[2] :].copy_(value)


@_write_kv_tail.register_fake
def _fake_write_kv_tail(cache_k, cache_v, key, value):
    return None


def _attention_with_owned_cache(
    self,
    positions,
    hidden_states,
    past_key_values,
    replace_position,
    use_cache,
    attention_mask,
):
    """Upstream LLaDA2 attention arithmetic; only the KV storage update differs."""
    if hidden_states.shape[0] == 0:
        return hidden_states
    batch, query_length, _ = hidden_states.size()
    qkv, _ = self.query_key_value(hidden_states)
    query, key, value = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    if self.use_qk_norm:
        query, key = self._apply_qk_norm(query, key)
    query, key = self.rotary_emb(
        positions.flatten(),
        query.flatten(0, 1),
        key.flatten(0, 1),
        fused_set_kv_buffer_arg=None,
    )
    query = query.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
    cache_k, cache_v = past_key_values
    _write_kv_tail(cache_k, cache_v, key, value)
    key, value = cache_k, cache_v
    present = (key, value) if use_cache else None
    key, value = self._apply_repeat(key, value)
    if attention_mask is not None and len(attention_mask.shape) == 3:
        attention_mask = attention_mask.unsqueeze(1)
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    output = output.transpose(1, 2).contiguous().reshape(batch, query_length, -1)
    output, _ = self.dense(output)
    return output, present


def prepare_attention(engine):
    """Graph KV inputs are owned scratch; eager KV inputs need a private copy."""
    for layer in engine.model.model.layers:
        attention = layer.attention
        if getattr(attention, "_owned_kv_storage_installed", False):
            continue
        original = attention.forward

        def forward(
            self,
            positions,
            hidden_states,
            past_key_values=None,
            replace_position=None,
            use_cache=None,
            attention_mask=None,
            _original=original,
        ):
            if past_key_values is None:
                return _original(
                    positions,
                    hidden_states,
                    past_key_values,
                    replace_position,
                    use_cache,
                    attention_mask,
                )
            if getattr(engine, "_eager_kv_storage", False):
                # contiguous() does not detach an already contiguous view from
                # the full multi-layer backing storage. clone() must own it.
                past_key_values = tuple(
                    value.clone(memory_format=torch.contiguous_format)
                    for value in past_key_values
                )
            return _attention_with_owned_cache(
                self,
                positions,
                hidden_states,
                past_key_values,
                replace_position,
                use_cache,
                attention_mask,
            )

        attention.forward = MethodType(forward, attention)
        attention._owned_kv_storage_installed = True


def install_eager_storage(engine):
    runner = engine.model_runner
    original = type(runner).forward_normal.__get__(runner, type(runner))

    @torch.no_grad()
    def normal(*args, **kwargs):
        engine._eager_kv_storage = True
        try:
            return original(*args, **kwargs)
        finally:
            engine._eager_kv_storage = False

    runner.forward_normal = normal
