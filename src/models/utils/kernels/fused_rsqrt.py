"""Fused Rsqrt kernel.

Pattern: 1 / sqrt(x)
Fuses: sqrt + reciprocal into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x):
    return torch.rsqrt(x)


# --- KERNEL ---
@triton.jit
def _rsqrt_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / tl.sqrt(x)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _rsqrt_bwd(X, Y, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    dx = -0.5 * dy * y * y * y
    tl.store(DX + offs, dx, mask=mask)


class FusedRsqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.rsqrt(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dx = -0.5 * dy * torch.rsqrt(x.pow(3))
        return dx


def kernel_fn(x):
    return FusedRsqrt.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
