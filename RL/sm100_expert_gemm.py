"""Autograd BF16 grouped GEMMs for Blackwell GPUs.

Expert padding is removed from projection outputs. Weight gradients accumulate
in float32 and are cast after reduction."""
import torch


def _layout(offsets, total_rows):
    layout = getattr(offsets, '_rl_sm100_layout', None)
    if layout is None:
        counts = offsets[1:] - offsets[:-1]
        padded = ((counts + 127) // 128).clamp_min(1) * 128
        ends = padded.cumsum(0).to(torch.int32)
        starts = torch.cat((ends.new_zeros(1), ends[:-1])).long()
        shift = torch.repeat_interleave(starts - offsets[:-1], counts,
                                        output_size=total_rows)
        positions = torch.arange(total_rows, device=offsets.device) + shift
        # The grouped kernels consume GPU prefix sums. Reserve a static upper
        # bound so neither allocation nor GEMM scheduling needs a host read.
        # Even an empty expert gets 128 zero rows for an exact zero DW.
        capacity = ((total_rows + 128 * counts.numel() + 127) // 128) * 128
        from moe_tensor_ops import padding_map
        layout = positions, ends, capacity, padding_map(offsets, ends, capacity)
        offsets._rl_sm100_layout = layout
    return layout


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, offsets):
        from sm100_backend import get_backend
        ops = get_backend()
        from moe_tensor_ops import matrix_map
        positions, ends, capacity, pad_map = _layout(offsets, x.shape[0])
        packed = matrix_map(x, pad_map)
        output = x.new_empty((capacity, weight.shape[1]))
        ops.m_grouped_bf16_gemm_nt_contiguous(packed, weight, output, ends, use_psum_layout=True)
        ctx.save_for_backward(packed, weight, positions, ends, pad_map)
        return matrix_map(output, positions)

    @staticmethod
    def backward(ctx, grad_output):
        from sm100_backend import get_backend
        ops = get_backend()
        from moe_tensor_ops import matrix_map
        packed, weight, positions, ends, pad_map = ctx.saved_tensors
        grad_packed = matrix_map(grad_output.contiguous(), pad_map)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx_packed = torch.empty_like(packed)
            ops.m_grouped_bf16_gemm_nn_contiguous(grad_packed, weight, dx_packed, ends, use_psum_layout=True)
            dx = matrix_map(dx_packed, positions)
        if ctx.needs_input_grad[1]:
            # Tensor-core accumulation remains FP32; cast once in the kernel's
            # epilogue, avoiding a second full-size weight-gradient buffer.
            dw = torch.empty_like(weight)
            ops.k_grouped_bf16_gemm_tn_contiguous(grad_packed, packed, dw, None, ends, use_psum_layout=True)
        return dx, dw, None


def grouped_linear(x, weight, offsets):
    if not isinstance(weight, torch.Tensor) or weight.ndim != 3:
        raise TypeError('SM100 grouped GEMM requires fused 3D expert parameters')
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError('SM100 grouped GEMM requires BF16 activations and weights')
    return _GroupedLinear.apply(x.contiguous(), weight, offsets)


def install_sm100_expert_gemm(model, *, verify=False):
    import sys
    from sm100_backend import get_backend
    get_backend()
    modules = {sys.modules[layer.__class__.__module__] for layer in model.modules()
               if layer.__class__.__name__ == 'LLaDA2MoeSparseMoeBlock'}
    for module in modules:
        original = module.triton_grouped_linear
        checked = [False]
        def wrapped(x, weight, offsets, _original=original, _checked=checked):
            result = grouped_linear(x, weight, offsets)
            if verify and not _checked[0]:
                with torch.no_grad():
                    reference = _original(x, weight, offsets)
                    relative = (result.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)
                    print(f'[SM100 GEMM] real-layer forward shape={tuple(x.shape)} relative_l2={float(relative):.6g}', flush=True)
                    if float(relative) > .01:
                        raise RuntimeError('SM100 grouped GEMM differs from reference')
                _checked[0] = True
            return result
        module.triton_grouped_linear = wrapped
    return len(modules)
