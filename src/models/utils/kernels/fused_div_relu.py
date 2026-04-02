"""Fused Div + ReLU kernel.

Pattern: y = relu(x / other) — fused division and ReLU activation.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, other):
    return torch.relu(x / other)


@triton.jit
def _div_relu_fwd(X, Other, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    ov = tl.load(Other + offs, mask=mask).to(tl.float32)
    d = xv / tl.where(ov == 0, 1e-12, ov)
    yv = tl.where(d > 0, d, 0.0)
    tl.store(Y + offs, yv, mask=mask)


@triton.jit
def _div_relu_bwd(X, Other, Y, DY, DX, DO, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    ov = tl.load(Other + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    ov_safe = tl.where(ov == 0, 1e-12, ov)
    d = xv / ov_safe
    gate = tl.where(d > 0, 1.0, 0.0)
    tl.store(DX + offs, dy * gate / ov_safe, mask=mask)
    tl.store(DO + offs, -dy * gate * xv / (ov_safe * ov_safe), mask=mask)


class FusedDivReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, other):
        assert x.is_contiguous() and other.is_contiguous()
        ctx.save_for_backward(x, other)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _div_relu_fwd[grid](x, other, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, other = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        do = torch.empty_like(other)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _div_relu_bwd[grid](x, other, ctx.needs_input_grad[0], dy, dx, do, N, BLOCK=BLOCK, num_warps=4)
        return dx, do


def kernel_fn(x, other):
    return FusedDivReLU.apply(x, other)


def can_use_kernel(x, other):
    return (x.is_cuda and other.is_cuda and
            x.is_contiguous() and other.is_contiguous() and
            x.shape == other.shape and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096), "other": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120), "other": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536), "other": (8, 256, 1536)},
}
