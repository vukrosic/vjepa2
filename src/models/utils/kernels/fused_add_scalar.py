"""Fused add scalar kernel.

Pattern: y = x + scalar
Fuses: scalar addition into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, scalar):
    return x + scalar


# --- KERNEL ---
@triton.jit
def _fused_add_scalar_fwd(X, Y, N: tl.constexpr, SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = x + SCALAR
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_add_scalar_bwd(DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    tl.store(DX + offs, dy, mask=mask)


class FusedAddScalar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scalar):
        assert x.is_contiguous()
        ctx.scalar = scalar
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_add_scalar_fwd[grid](x, y, N, scalar, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        N = dy.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_add_scalar_bwd[grid](dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx, None


def kernel_fn(x, scalar):
    return FusedAddScalar.apply(x, scalar)


def can_use_kernel(x, scalar):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            isinstance(scalar, (int, float)))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "scalar": 1e-5},
    "vit_h":  {"x": (2, 2048, 5120), "scalar": 1e-4},
    "small":  {"x": (8, 256, 1536), "scalar": 0.01},
}
