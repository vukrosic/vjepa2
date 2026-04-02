"""Fused Clamp kernel.
Pattern: y = clamp(x, min_val, max_val)
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl

def baseline_fn(x, min_val, max_val):
    return torch.clamp(x, min_val, max_val)

@triton.jit
def _clamp_fwd(X, Y, N: tl.constexpr, MIN_V: tl.constexpr, MAX_V: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N: break
        v = tl.load(X + offs).to(tl.float32)
        y = MIN_V if v < MIN_V else (MAX_V if v > MAX_V else v)
        tl.store(Y + offs, y)

@triton.jit
def _clamp_bwd(X, DY, DX, N: tl.constexpr, MIN_V: tl.constexpr, MAX_V: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N: break
        x = tl.load(X + offs).to(tl.float32)
        dy = tl.load(DY + offs).to(tl.float32)
        dx = dy if (x >= MIN_V and x <= MAX_V) else 0.0
        tl.store(DX + offs, dx)

class FusedClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.min_val = float(min_val)
        ctx.max_val = float(max_val)
        y = torch.empty_like(x)
        N = x.numel(); BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _clamp_fwd[grid](x, y, N, float(min_val), float(max_val), BLOCK=BLOCK, num_warps=4)
        return y
    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel(); BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _clamp_bwd[grid](x, dy, dx, N, ctx.min_val, ctx.max_val, BLOCK=BLOCK, num_warps=4)
        return dx, None, None

def kernel_fn(x, min_val, max_val):
    return FusedClamp.apply(x, min_val, max_val)

def can_use_kernel(x, min_val, max_val):
    return x.is_cuda and x.is_contiguous() and x.dtype in (torch.float16, torch.float32, torch.bfloat16)

SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
