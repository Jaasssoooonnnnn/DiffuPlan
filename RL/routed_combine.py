"""Fuse expert unpermutation, weighting and reduction without full temporaries."""
import torch
import triton
import triton.language as tl
_REPORTED = False


@triton.jit
def _forward(X, W, Inv, Y, T, H:tl.constexpr, K:tl.constexpr,
             LOW_WEIGHT:tl.constexpr, BT:tl.constexpr=16, BH:tl.constexpr=128):
    t=tl.program_id(0)*BT+tl.arange(0,BT)
    h=tl.program_id(1)*BH+tl.arange(0,BH)
    value=tl.full((BT,BH),0,tl.float32)
    for k in range(K):
        row=tl.load(Inv+t*K+k,t<T,0)
        x=tl.load(X+row[:,None]*H+h[None,:],(t[:,None]<T)&(h[None,:]<H),0).to(tl.float32)
        w=tl.load(W+t*K+k,t<T,0).to(tl.float32)
        product=x*w[:,None]
        if LOW_WEIGHT: product=product.to(X.dtype.element_ty).to(tl.float32)
        value=value+product
    tl.store(Y+t[:,None]*H+h[None,:],value,(t[:,None]<T)&(h[None,:]<H))


@triton.jit
def _backward_x(DY, W, Order, DX, M,H:tl.constexpr,K:tl.constexpr,
                BR:tl.constexpr=16,BH:tl.constexpr=128):
    row=(tl.program_id(0)*BR+tl.arange(0,BR)).to(tl.int64)
    h=tl.program_id(1)*BH+tl.arange(0,BH)
    original=tl.load(Order+row,row<M,0)
    dy=tl.load(DY+(original//K)[:,None]*H+h[None,:],(row[:,None]<M)&(h[None,:]<H),0).to(tl.float32)
    w=tl.load(W+original,row<M,0).to(tl.float32)
    tl.store(DX+row[:,None]*H+h[None,:],dy*w[:,None],(row[:,None]<M)&(h[None,:]<H))


@triton.jit
def _backward_w(X,DY,Order,DW,M,H:tl.constexpr,K:tl.constexpr,
                LOW_WEIGHT:tl.constexpr,BR:tl.constexpr=4,BH:tl.constexpr=2048):
    row=(tl.program_id(0)*BR+tl.arange(0,BR)).to(tl.int64)
    h=tl.arange(0,BH)
    original=tl.load(Order+row,row<M,0)
    x=tl.load(X+row[:,None]*H+h[None,:],(row[:,None]<M)&(h[None,:]<H),0).to(tl.float32)
    dy=tl.load(DY+(original//K)[:,None]*H+h[None,:],(row[:,None]<M)&(h[None,:]<H),0).to(tl.float32)
    product=x*dy
    if LOW_WEIGHT: product=product.to(X.dtype.element_ty).to(tl.float32)
    result=tl.sum(product,1)
    tl.store(DW+original,result,row<M)


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,order,weights):
        t,k=weights.shape;h=x.shape[1]
        inverse=torch.empty_like(order)
        inverse[order]=torch.arange(order.numel(),device=order.device,dtype=order.dtype)
        output=x.new_empty((t,h))
        _forward[(triton.cdiv(t,16),triton.cdiv(h,128))](x,weights,inverse,output,t,h,k,
            weights.dtype!=torch.float32,num_warps=8,enable_fp_fusion=False)
        ctx.save_for_backward(x,order,weights)
        return output

    @staticmethod
    def backward(ctx,dy):
        x,order,w=ctx.saved_tensors;dy=dy.contiguous();m,h=x.shape;k=w.shape[1]
        dx=dw=None
        if ctx.needs_input_grad[0]:
            dx=torch.empty_like(x)
            _backward_x[(triton.cdiv(m,16),triton.cdiv(h,128))](dy,w,order,dx,m,h,k,
                num_warps=8,enable_fp_fusion=False)
        if ctx.needs_input_grad[2]:
            dw=torch.empty_like(w)
            _backward_w[(triton.cdiv(m,4),)](x,dy,order,dw,m,h,k,w.dtype!=torch.float32,
                BH=triton.next_power_of_2(h),num_warps=8,enable_fp_fusion=False)
        return dx,None,dw


def combine(sorted_output,order,weights):
    global _REPORTED
    if not _REPORTED and sorted_output.is_cuda:
        import os
        if os.environ.get('RL_REPORT_KERNEL_DTYPES') == '1':
            print(f'[Fused route] activations={sorted_output.dtype} router_weights={weights.dtype}', flush=True)
        _REPORTED = True
    if sorted_output.is_cuda and sorted_output.dtype==torch.bfloat16 and weights.dtype in (torch.bfloat16,torch.float32):
        return _Combine.apply(sorted_output,order,weights)
    restored=torch.empty_like(sorted_output)
    restored[order]=sorted_output
    return (restored.view(*weights.shape,-1)*weights.unsqueeze(-1)).sum(1).to(sorted_output.dtype)
