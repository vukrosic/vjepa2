"""Fused Swish kernel with beta parameter.

Pattern: y = x * sigmoid(beta * x)
Fuses: scale + sigmoid + multiply into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, beta=1.0):
    return x * torch.sigmoid(beta * x)


@triton.jit
def _swish_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, BETA: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-BETA * x))
    y = x * sig
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _swish_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr, BETA: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-BETA * x))
    dsig = sig * (1.0 - sig)
    dx = dy * (sig + BETA * x * dsig)
    tl.store(DX + offs, dx, mask=mask)


class FusedSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta=1.0):
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _swish_fwd[(n_blocks,)](x, y, N, BLOCK=BLOCK, BETA=beta, num_warps=4)
        ctx.save_for_backward(x)
        ctx.beta = beta
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _swish_bwd[(n_blocks,)](x, dy, dx, N, BLOCK=BLOCK, BETA=ctx.beta, num_warps=4)
        return dx, None


def kernel_fn(x, beta=1.0):
    return FusedSwish.apply(x, beta)


def can_use_kernel(x, beta=1.0):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
