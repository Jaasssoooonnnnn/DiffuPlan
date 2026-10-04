"""Share compact KV storage across bounded batch/cache CUDA graph shapes."""

from contextlib import contextmanager
import torch
from types import SimpleNamespace


@contextmanager
def bounded_graph_capture(max_kv_tokens):
    from dinfer.decoding.diffusion_runner import CudaGraphRunner

    original_capture = CudaGraphRunner.capture
    original_one = CudaGraphRunner.capture_one_batch_size

    def capture(self):
        old = self.past_key_values
        layers, _, _, heads, _, dim = old.shape
        del self.past_key_values
        device, dtype = old.device, old.dtype
        del old
        storage = torch.zeros(
            layers * 2 * max_kv_tokens * heads * dim, device=device, dtype=dtype
        )
        self._bounded_kv_views = {}
        for capacity in self.cache_lengths:
            max_bs = min(self.max_bs, max_kv_tokens // capacity)
            elements = layers * 2 * max_bs * heads * capacity * dim
            self._bounded_kv_views[capacity] = storage[:elements].view(
                layers, 2, max_bs, heads, capacity, dim
            )
        self.past_key_values = self._bounded_kv_views[max(self.cache_lengths)]
        max_q = max(2 * self.model_runner.block_length, max(self.prefill_lengths))
        self.attention_mask = torch.ones(
            (self.max_bs, max_q, self.model_runner.max_length),
            device=device,
            dtype=torch.bool,
        )
        self.attention_mask[0, 0, 0] = False
        original_capture(self)
        invalid = [key for key, graph in self.graphs.items() if graph is None]
        for key in invalid:
            del self.graphs[key]
            del self.output_buffers[key]

    def one(
        self,
        bs,
        forward,
        is_decode_phase=True,
        length=0,
        cache_length=0,
        use_mask=False,
    ):
        if is_decode_phase:
            if bs * cache_length > max_kv_tokens:
                return None, None
            self.past_key_values = self._bounded_kv_views[cache_length]

        def shared_forward(**kwargs):
            output = forward(**kwargs)
            hidden = output.logits
            if not hasattr(self, "_shared_graph_hidden"):
                max_q = max(
                    2 * self.model_runner.block_length, max(self.prefill_lengths)
                )
                self._shared_graph_hidden = torch.empty(
                    (self.max_bs * max_q, hidden.shape[-1]),
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            shared_hidden = self._shared_graph_hidden[: bs * length].view(
                bs, length, -1
            )
            shared_hidden.copy_(hidden)
            if is_decode_phase:
                target = self._bounded_kv_views[cache_length]
                values = []
                for i, value in enumerate(output.past_key_values):
                    destination = target[i // 2, i % 2, :bs]
                    destination[:, :, -length:].copy_(value[:, :, -length:])
                    values.append(destination)
            else:
                capacity = min(c for c in self.cache_lengths if c >= length)
                target = self._bounded_kv_views[capacity]
                values = []
                for i, value in enumerate(output.past_key_values):
                    destination = target[i // 2, i % 2, :bs, :, :length]
                    destination.copy_(value)
                    values.append(destination)
            # Return views of common storage. Intermediate full-cache outputs
            # die inside run_once, allowing graph pools to reuse their workspace.
            return SimpleNamespace(
                logits=shared_hidden,
                past_key_values=values,
                hidden_states=shared_hidden,
            )

        return original_one(
            self, bs, shared_forward, is_decode_phase, length, cache_length, use_mask
        )

    CudaGraphRunner.capture = capture
    CudaGraphRunner.capture_one_batch_size = one
    try:
        yield
    finally:
        CudaGraphRunner.capture = original_capture
        CudaGraphRunner.capture_one_batch_size = original_one
