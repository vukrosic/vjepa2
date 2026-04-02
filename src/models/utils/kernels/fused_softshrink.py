"""Fused SoftShrink kernel.

Pattern: y = x - lambd if x > lambd, else x + lambd if x < -lambd, else 0
Fuses: comparison + arithmetic into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, lambd=0.5):
    return torch.nn.functional.softshrink(x, lambd=lambd)


@triton.jit
def _fused_softshrink_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, lambd: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.where(v > lambd, v - lambd, tl.where(v < -lambd, v + lambd, 0.0))
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_softshrink_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr, lambd: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    dx = tl.where((x > -lambd) & (x < lambd), 0.0, dy)
    tl.store(DX + offs, dx, mask=mask)


class FusedSoftShrink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd=0.5):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.lambd = lambd
        N = x.numel()
        y = torch.empty_like(x)
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_softshrink_fwd[grid](x, y, N, BLOCK=BLOCK, lambd=lambd, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_softshrink_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, lambd=ctx.lambd, num_warps=4)
        return dx, None


def kernel_fn(x, lambd=0.5):
    return FusedSoftShrink.apply(x, lambd)


def can_use_kernel(x, lambd=0.5):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536)},
}
