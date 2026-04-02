"""Fused LeakyReLU kernel.

Source: src/models/utils/modules.py (various activation uses)
Pattern: torch.nn.functional.leaky_relu(x, negative_slope=0.01)
Fuses: comparison + multiply + select into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, negative_slope=0.01):
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)


# --- KERNEL ---
@triton.jit
def _leaky_relu_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, neg_slope: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(x > 0, x, x * neg_slope)
    tl.store(Y + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _leaky_relu_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr, neg_slope: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    dx = tl.where(x > 0, dy, dy * neg_slope)
    tl.store(DX + offs, dx.to(x.dtype), mask=mask)


class FusedLeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, negative_slope=0.01):
        x_c = x.contiguous()
        y = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _leaky_relu_fwd[(n_blocks,)](x_c, y, N, BLOCK=BLOCK, neg_slope=negative_slope, num_warps=4)
        ctx.save_for_backward(x_c)
        ctx.neg_slope = negative_slope
        return y

    @staticmethod
    def backward(ctx, dy):
        (x_c,) = ctx.saved_tensors
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _leaky_relu_bwd[(n_blocks,)](x_c, dy_c, dx, N, BLOCK=BLOCK, neg_slope=ctx.neg_slope, num_warps=4)
        return dx, None


def kernel_fn(x, negative_slope=0.01):
    return FusedLeakyReLU.apply(x, negative_slope)


def can_use_kernel(x, negative_slope=0.01):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l_mlp":  {"x": (2, 1024, 4096)},
    "vit_h_mlp":  {"x": (2, 2048, 5120)},
    "small":      {"x": (8, 256, 1536)},
}
