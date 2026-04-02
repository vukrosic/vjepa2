"""Fused power with scalar kernel.

Pattern: y = x ** exponent
Fuses: pow into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, exponent):
    return x ** exponent


# --- KERNEL ---
@triton.jit
def _fused_pow_scalar_fwd(X, Y, N: tl.constexpr, EXP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.pow(x, EXP)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_pow_scalar_bwd(X, DY, DX, N: tl.constexpr, EXP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    dx = dy * EXP * tl.pow(x, EXP - 1.0)
    tl.store(DX + offs, dx, mask=mask)


class FusedPowScalar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exponent):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.exponent = float(exponent)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_pow_scalar_fwd[grid](x, y, N, float(exponent), BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_pow_scalar_bwd[grid](x, dy, dx, N, ctx.exponent, BLOCK=BLOCK, num_warps=4)
        return dx, None


def kernel_fn(x, exponent):
    return FusedPowScalar.apply(x, exponent)


def can_use_kernel(x, exponent):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            isinstance(exponent, (int, float)))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "exponent": 2.0},
    "vit_h":  {"x": (2, 2048, 5120), "exponent": 0.5},
    "small":  {"x": (8, 256, 1536), "exponent": 3.0},
}
