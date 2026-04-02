"""Fused Log + Abs kernel.

Pattern: y = log(abs(x)) — fused absolute value and logarithm.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.log(tl.where(x.abs() <= 0, 1e-12, x.abs()))


@triton.jit
def _log_abs_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    av = tl.abs(xv)
    tl.store(Y + offs, tl.log(tl.where(av <= 0, 1e-12, av)), mask=mask)


@triton.jit
def _log_abs_bwd(X, Y, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    av = tl.abs(xv)
    av_safe = tl.where(av <= 0, 1e-12, av)
    dabs = tl.where(xv >= 0, 1.0, -1.0)
    dx = dy * (1.0 / av_safe) * dabs
    tl.store(DX + offs, dx, mask=mask)


class FusedLogAbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _log_abs_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _log_abs_bwd[grid](x, ctx.needs_input_grad[0], dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedLogAbs.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
