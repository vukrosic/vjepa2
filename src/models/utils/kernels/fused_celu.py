"""Fused CELU (Continuously Differentiable Exponential Linear Unit) kernel.

Pattern: y = x if x >= 0 else alpha * (exp(x / alpha) - 1)
Fuses: comparison + exp + divide + multiply into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, alpha=1.0):
    return torch.nn.functional.celu(x, alpha=alpha)


@triton.jit
def _fused_celu_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, alpha: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.where(v > 0, v, alpha * (tl.exp(tl.minimum(v / alpha, 20.0)) - 1.0))
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_celu_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr, alpha: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    dx = tl.where(x > 0, dy, dy * tl.exp(tl.minimum(x / alpha, 20.0)))
    tl.store(DX + offs, dx, mask=mask)


class FusedCelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        N = x.numel()
        y = torch.empty_like(x)
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_celu_fwd[grid](x, y, N, BLOCK=BLOCK, alpha=alpha, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_celu_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, alpha=ctx.alpha, num_warps=4)
        return dx, None


def kernel_fn(x, alpha=1.0):
    return FusedCelu.apply(x, alpha)


def can_use_kernel(x, alpha=1.0):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536)},
}
