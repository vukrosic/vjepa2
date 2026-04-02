"""Fused hardsigmoid kernel.

Pattern: torch.nn.functional.hardsigmoid(x)
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.hardsigmoid(x)


@triton.jit
def _fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(v <= -3.0, 0.0, tl.where(v >= 3.0, 1.0, (v + 3.0) / 6.0))
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    dx = tl.where((v > -3.0) & (v < 3.0), dy / 6.0, 0.0)
    tl.store(DX + offs, dx, mask=mask)


class Hardsigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return Hardsigmoid.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
