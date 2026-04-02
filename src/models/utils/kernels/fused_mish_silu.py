"""Fused mish_silu kernel.

Pattern: torch.nn.functional.mish(torch.nn.functional.silu(x))
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.mish(torch.nn.functional.silu(x))


@triton.jit
def _fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-v, 20.0)))
    sil = v * sig
    sp = tl.log(1.0 + tl.exp(tl.minimum(sil, 20.0)))
    e2sp = tl.exp(tl.minimum(2.0 * sp, 40.0))
    tanh_sp = (e2sp - 1.0) / (e2sp + 1.0)
    y = sil * tanh_sp
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-v, 20.0)))
    dsig = sig * (1.0 - sig)
    dsil = sig + v * dsig
    e2sp = tl.exp(tl.minimum(2.0 * sp, 40.0))
    tanh_sp = (e2sp - 1.0) / (e2sp + 1.0)
    sig_sp = 1.0 / (1.0 + tl.exp(tl.minimum(-sil, 20.0)))
    dtanh_sp = 1.0 - tanh_sp * tanh_sp
    dsp_dsil = sig_sp
    d_out_dv = dsil * (tanh_sp + sil * dtanh_sp * dsp_dsil)
    dx = dy * d_out_dv
    tl.store(DX + offs, dx, mask=mask)


class MishSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(sil, sp, v)
        return y

    @staticmethod
    def backward(ctx, dy):
        sil, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(sil)
        N = sil.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _bwd[grid](sil, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return MishSilu.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
