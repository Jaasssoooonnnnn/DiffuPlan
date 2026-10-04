"""Vectorized MoE padding copies and SiLU product with BF16 rounding preserved."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _pad_map(Offsets,Ends,Map,C,E:tl.constexpr,ITER:tl.constexpr,B:tl.constexpr=1024):
    row=tl.program_id(0)*B+tl.arange(0,B)
    lo=tl.full((B,),0,tl.int32);hi=tl.full((B,),E,tl.int32)
    for _ in range(ITER):
        mid=(lo+hi)//2
        boundary=tl.load(Ends+mid,mid<E,2147483647)
        before=row<boundary
        hi=tl.where(before,mid,hi);lo=tl.where(before,lo,mid+1)
    group=lo
    start=tl.load(Offsets+group,group<=E,0)
    end=tl.load(Offsets+group+1,group<E,0)
    padded_start=tl.load(Ends+group-1,group>0,0)
    local=row-padded_start
    valid=(group<E)&(local<end-start)
    tl.store(Map+row,tl.where(valid,start+local,-1),row<C)


@triton.jit
def _matrix_map(X,Map,Y,N,H:tl.constexpr,B:tl.constexpr=8192):
    i=(tl.program_id(0)*B+tl.arange(0,B)).to(tl.int64)
    row=i//H;col=i%H
    source=tl.load(Map+row,row<N,-1)
    value=tl.load(X+source*H+col,(row<N)&(source>=0),0)
    tl.store(Y+i,value,i<N*H)


def padding_map(offsets,ends,capacity):
    result=torch.empty(capacity,device=offsets.device,dtype=torch.long)
    experts=ends.numel()
    _pad_map[(triton.cdiv(capacity,1024),)](offsets,ends,result,capacity,experts,
        (experts+1).bit_length(),num_warps=4)
    return result


def matrix_map(x,mapping):
    out=x.new_empty((mapping.numel(),x.shape[1]))
    if out.numel():
        _matrix_map[(triton.cdiv(out.numel(),8192),)](x,mapping,out,out.shape[0],out.shape[1],num_warps=8)
    return out


@triton.jit
def _swiglu_forward(X,Y,N,H:tl.constexpr,B:tl.constexpr=4096):
    i=(tl.program_id(0)*B+tl.arange(0,B)).to(tl.int64)
    row=i//H;col=i%H
    gate=tl.load(X+row*(2*H)+col,i<N,0).to(tl.float32)
    up=tl.load(X+row*(2*H)+H+col,i<N,0).to(tl.float32)
    activated=(gate*tl.sigmoid(gate)).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y+i,activated*up,i<N)


@triton.jit
def _swiglu_backward(X,DY,DX,N,H:tl.constexpr,B:tl.constexpr=4096):
    i=(tl.program_id(0)*B+tl.arange(0,B)).to(tl.int64)
    row=i//H;col=i%H
    gate=tl.load(X+row*(2*H)+col,i<N,0).to(tl.float32)
    up=tl.load(X+row*(2*H)+H+col,i<N,0).to(tl.float32)
    dy=tl.load(DY+i,i<N,0).to(tl.float32)
    sigmoid=tl.sigmoid(gate)
    activated=(gate*sigmoid).to(X.dtype.element_ty).to(tl.float32)
    dactivated=(dy*up).to(X.dtype.element_ty).to(tl.float32)
    dgate=dactivated*sigmoid*(1+gate*(1-sigmoid))
    tl.store(DX+row*(2*H)+col,dgate,i<N)
    tl.store(DX+row*(2*H)+H+col,dy*activated,i<N)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        h=x.shape[1]//2;y=x.new_empty((x.shape[0],h));ctx.save_for_backward(x)
        if y.numel():_swiglu_forward[(triton.cdiv(y.numel(),4096),)](x,y,y.numel(),h,num_warps=8,enable_fp_fusion=False)
        return y
    @staticmethod
    def backward(ctx,dy):
        (x,)=ctx.saved_tensors;dx=torch.empty_like(x);dy=dy.contiguous();h=x.shape[1]//2
        if dy.numel():_swiglu_backward[(triton.cdiv(dy.numel(),4096),)](x,dy,dx,dy.numel(),h,num_warps=8,enable_fp_fusion=False)
        return dx


def swiglu(x):
    if x.is_cuda and x.dtype==torch.bfloat16:
        return _SwiGLU.apply(x)
    h=x.shape[1]//2
    return F.silu(x[:,:h])*x[:,h:]
