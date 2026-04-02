"""Fused Swish v2 kernel.

Pattern: y = x * sigmoid(beta * x) (beta-swish, beta=1 default)
Fuses: scale + sigmoid + multiply into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, beta=1.0):
    return torch.nn.functional.silu(x) if beta == 1.0 else x * torch.sigmoid(beta * x)


# --- KERNEL ---
@triton.jit
def _fused_swish_v2_fwd(X, Y, N: tl.constexpr, BETA: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-BETA * x))
    y = x * sig
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_swish_v2_bwd(X, DY, DX, N: tl.constexpr, BETA: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-BETA * x))
    dsig = sig * (1.0 - sig)
    dx = dy * (sig + BETA * x * dsig)
    tl.store(DX + offs, dx, mask=mask)


class FusedSwishV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta=1.0):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.beta = beta
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_swish_v2_fwd[grid](x, y, N, beta, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_swish_v2_bwd[grid](x, dy, dx, N, ctx.beta, BLOCK=BLOCK, num_warps=4)
        return dx, None


def kernel_fn(x, beta=1.0):
    return FusedSwishV2.apply(x, beta)


def can_use_kernel(x, beta=1.0):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "beta": 1.0},
    "vit_h":  {"x": (2, 2048, 5120), "beta": 1.0},
    "small":  {"x": (8, 256, 1536), "beta": 1.0},
}
