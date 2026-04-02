"""Fused Neg + SiLU kernel.

Pattern: y = silu(-x) — fused negation and SiLU activation.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.silu(-x)


@triton.jit
def _neg_silu_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    nv = -xv
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-nv, 20.0)))
    tl.store(Y + offs, nv * sig, mask=mask)


@triton.jit
def _neg_silu_bwd(X, Y, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    nv = -xv
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-nv, 20.0)))
    dsig = sig * (1.0 - sig)
    # d(silu(-x))/dx = (-1) * (sig + nv * dsig) = -sig + nv * dsig
    dx = dy * (-sig + nv * dsig)
    tl.store(DX + offs, dx, mask=mask)


class FusedNegSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _neg_silu_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _neg_silu_bwd[grid](x, ctx.needs_input_grad[0], dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedNegSiLU.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
