"""Fused abs kernel.

Pattern: y = |x|
Fuses: comparison + select into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return x.abs()


@triton.jit
def _abs_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(x >= 0, x, -x)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _abs_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    dx = tl.where(x >= 0, dy, -dy)
    tl.store(DX + offs, dx, mask=mask)


class FusedAbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _abs_fwd[(n_blocks,)](x, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _abs_bwd[(n_blocks,)](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedAbs.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
