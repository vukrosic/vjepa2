"""Fused Div-Scalar kernel.

Pattern: x / scalar — divides tensor by a scalar.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, scalar):
    return x / scalar


@triton.jit
def _div_scalar_fwd(X, Y, N: tl.constexpr, SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    tl.store(Y + offs, xv / SCALAR, mask=mask)


@triton.jit
def _div_scalar_bwd(DY, DX, N: tl.constexpr, SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    tl.store(DX + offs, dy / SCALAR, mask=mask)


class FusedDivScalar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scalar):
        assert x.is_contiguous()
        ctx.scalar = float(scalar)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _div_scalar_fwd[grid](x, y, N, float(scalar), BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        N = dy.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _div_scalar_bwd[grid](dy, dx, N, ctx.scalar, BLOCK=BLOCK, num_warps=4)
        return dx, None


def kernel_fn(x, scalar):
    return FusedDivScalar.apply(x, scalar)


def can_use_kernel(x, scalar=None):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "scalar": 2.0},
    "vit_h":  {"x": (2, 2048, 5120), "scalar": 0.5},
    "small":  {"x": (8, 256, 1536), "scalar": 1.5},
}
