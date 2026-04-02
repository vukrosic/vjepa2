"""Fused LeakyReLU v2 activation kernel.

Pattern: y = x if x >= 0 else negative_slope * x
Alternative implementation.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, negative_slope=0.01):
    return torch.nn.functional.leaky_relu(x, negative_slope)


# --- KERNEL ---
@triton.jit
def _fused_leaky_relu_v2_fwd(X, Y, N: tl.constexpr, NEG: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.where(x >= 0, x, x * NEG)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_leaky_relu_v2_bwd(X, DY, DX, N: tl.constexpr, NEG: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    dx = tl.where(x >= 0, dy, dy * NEG)
    tl.store(DX + offs, dx, mask=mask)


class FusedLeakyReLUV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, negative_slope=0.01):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.neg_slope = negative_slope
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_leaky_relu_v2_fwd[grid](x, y, N, negative_slope, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_leaky_relu_v2_bwd[grid](x, dy, dx, N, ctx.neg_slope, BLOCK=BLOCK, num_warps=4)
        return dx, None


def kernel_fn(x, negative_slope=0.01):
    return FusedLeakyReLUV2.apply(x, negative_slope)


def can_use_kernel(x, negative_slope=0.01):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "negative_slope": 0.01},
    "vit_h":  {"x": (2, 2048, 5120), "negative_slope": 0.02},
    "small":  {"x": (8, 256, 1536), "negative_slope": 0.1},
}
