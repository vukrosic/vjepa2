"""Fused log-sigmoid kernel.

Pattern: y = log(sigmoid(x)) = -softplus(-x)
Fuses log + sigmoid into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.logsigmoid(x)


@triton.jit
def _fused_logsigmoid_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask).to(tl.float32)
    # log(sigmoid(x)) = -softplus(-x) = log(1/(1+exp(-x))) = -log(1+exp(-x))
    # stable: -log(1 + exp(-|x|)) for x >= 0, and x - log(1+exp(x)) for x < 0
    # simplifies to: x - softplus(x) = x - log(exp(x)+1)
    # For negative x: -softplus(-x) = log(sigmoid(x)) = x - log(1+exp(x))
    # For positive x: -softplus(-x) = -log(1+exp(-x)) = log(exp(-x)/(1+exp(-x)))
    y = v - tl.log(tl.exp(tl.minimum(v, 20.0)) + 1.0)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_logsigmoid_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    # d/dx log(sigmoid(x)) = 1 - sigmoid(x) = exp(-x) / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-x, 20.0)))
    dx = dy * (1.0 - sig)
    tl.store(DX + offs, dx, mask=mask)


class FusedLogSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        N = x.numel()
        y = torch.empty_like(x)
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_logsigmoid_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_logsigmoid_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedLogSigmoid.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536)},
}
