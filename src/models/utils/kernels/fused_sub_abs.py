"""Fused Sub + Abs kernel.

Pattern: y = abs(x - other) — fused subtract and absolute value.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, other):
    return (x - other).abs()


@triton.jit
def _sub_abs_fwd(X, Other, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    ov = tl.load(Other + offs, mask=mask).to(tl.float32)
    tl.store(Y + offs, tl.abs(xv - ov), mask=mask)


@triton.jit
def _sub_abs_bwd(X, Other, DY, DX, DO, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    ov = tl.load(Other + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    d = xv - ov
    dabs = tl.where(d >= 0, 1.0, -1.0)
    tl.store(DX + offs, dy * dabs, mask=mask)
    tl.store(DO + offs, -dy * dabs, mask=mask)


class FusedSubAbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, other):
        assert x.is_contiguous() and other.is_contiguous()
        ctx.save_for_backward(x, other)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _sub_abs_fwd[grid](x, other, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, other = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        do = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _sub_abs_bwd[grid](x, other, dy, dx, do, N, BLOCK=BLOCK, num_warps=4)
        return dx, do


def kernel_fn(x, other):
    return FusedSubAbs.apply(x, other)


def can_use_kernel(x, other):
    return (x.is_cuda and other.is_cuda and
            x.is_contiguous() and other.is_contiguous() and
            x.shape == other.shape and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096), "other": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120), "other": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536), "other": (8, 256, 1536)},
}
