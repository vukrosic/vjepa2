"""Fused silog kernel.

Pattern: torch.nn.functional.silu(torch.log(torch.abs(x) + 1e-8))
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.silu(torch.log(torch.abs(x) + 1e-8))


@triton.jit
def _fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=1.0).to(tl.float32)
    logv = tl.log(tl.abs(v) + 1e-8)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-logv, 20.0)))
    y = logv * sig
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=1.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    logv = tl.log(tl.abs(v) + 1e-8)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-logv, 20.0)))
    dsig = sig * (1.0 - sig)
    dlogv_dv = 1.0 / (tl.abs(v) + 1e-8) * (1.0 if v >= 0 else -1.0)
    dx = dy * (dlogv_dv * sig + logv * dsig * dlogv_dv)
    tl.store(DX + offs, dx, mask=mask)


class Silog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(logv, v)
        return y

    @staticmethod
    def backward(ctx, dy):
        logv, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(logv)
        N = logv.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _bwd[grid](logv, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return Silog.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
