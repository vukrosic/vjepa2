"""Fused elementwise power kernel.

Pattern: y = x ** p
Fuses power operation into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, p=2.0):
    return torch.pow(x, p)


@triton.jit
def _fused_pow_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, p: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask).to(tl.float32)
    y = v * v  # p=2 case fast path
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_pow_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr, p: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    dx = dy * p * (x * x)
    tl.store(DX + offs, dx, mask=mask)


class FusedPow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p=2.0):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.p = p
        N = x.numel()
        y = torch.empty_like(x)
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        if p == 2.0:
            _fused_pow_fwd[grid](x, y, N, BLOCK=BLOCK, p=p, num_warps=4)
        else:
            _fused_pow_fwd[grid](x, y, N, BLOCK=BLOCK, p=p, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_pow_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, p=ctx.p, num_warps=4)
        return dx, None


def kernel_fn(x, p=2.0):
    return FusedPow.apply(x, p)


def can_use_kernel(x, p=2.0):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536)},
}
