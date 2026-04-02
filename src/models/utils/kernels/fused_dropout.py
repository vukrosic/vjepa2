"""Fused dropout kernel.

Pattern: torch.nn.functional.dropout(x, p=0.5, training=True)
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.dropout(x, p=0.5, training=True)


@triton.jit
def _fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, seed: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    keep = tl.rand(tl.rand(null, 0), seed=0) > 0.5
    y = tl.where(keep, v / 0.5, 0.0)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr, seed: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    keep = tl.rand(tl.rand(null, 0), seed=0) > 0.5
    dx = tl.where(keep, dy / 0.5, 0.0)
    tl.store(DX + offs, dx, mask=mask)


class Dropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        import triton
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        seed = tl.seed()
        _fwd[grid](x, y, N, BLOCK=BLOCK, seed=seed, num_warps=4)
        ctx.seed = seed
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _bwd[grid](x, dy, dx, N, BLOCK=BLOCK, seed=ctx.seed, num_warps=4)
        return dx


def kernel_fn(x):
    return Dropout.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
