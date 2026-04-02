"""Fused abs_neg kernel.

Pattern: -torch.abs(x)
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return -torch.abs(x)


@triton.jit
def _fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = -tl.abs(v)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    dx = -tl.sign(v) * dy
    tl.store(DX + offs, dx, mask=mask)


class AbsNeg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(v)
        return y

    @staticmethod
    def backward(ctx, dy):
        v, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(v)
        N = v.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _bwd[grid](v, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return AbsNeg.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
