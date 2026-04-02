"""Fused SiLU (Sigmoid Linear Unit) kernel.

Pattern: y = x * sigmoid(x)
Simple element-wise activation with pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.silu(x)


@triton.jit
def _fused_silu_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-v, 20.0)))
    y = v * sig
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_silu_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-x, 20.0)))
    dsig = sig * (1.0 - sig)
    dx = dy * (sig + x * dsig)
    tl.store(DX + offs, dx, mask=mask)


class FusedSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        N = x.numel()
        y = torch.empty_like(x)
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_silu_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_silu_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedSilu.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536)},
}
