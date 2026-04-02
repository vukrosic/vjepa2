"""Fused LeakyReLU kernel.
Pattern: y = x if x > 0 else negative_slope * x
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl

def baseline_fn(x, negative_slope=0.01):
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

@triton.jit
def _leaky_relu_fwd(X, Y, N: tl.constexpr, NEG_SLOPE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N: break
        v = tl.load(X + offs).to(tl.float32)
        y = v if v > 0 else v * NEG_SLOPE
        tl.store(Y + offs, y)

@triton.jit
def _leaky_relu_bwd(X, DY, DX, N: tl.constexpr, NEG_SLOPE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N: break
        x = tl.load(X + offs).to(tl.float32)
        dy = tl.load(DY + offs).to(tl.float32)
        dx = dy if x > 0 else dy * NEG_SLOPE
        tl.store(DX + offs, dx)

class FusedLeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, negative_slope=0.01):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.neg_slope = negative_slope
        y = torch.empty_like(x)
        N = x.numel(); BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _leaky_relu_fwd[grid](x, y, N, float(negative_slope), BLOCK=BLOCK, num_warps=4)
        return y
    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel(); BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _leaky_relu_bwd[grid](x, dy, dx, N, float(ctx.neg_slope), BLOCK=BLOCK, num_warps=4)
        return dx, None

def kernel_fn(x, negative_slope=0.01):
    return FusedLeakyReLU.apply(x, negative_slope)

def can_use_kernel(x, negative_slope=0.01):
    return x.is_cuda and x.is_contiguous() and x.dtype in (torch.float16, torch.float32, torch.bfloat16)

SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
