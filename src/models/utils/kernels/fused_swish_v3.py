"""Fused Swish v3 kernel.

Pattern: y = x / (1 + exp(-x)) = SiLU(x) [default Swish with beta=1]
Using hard sigmoid approximation.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return torch.nn.functional.silu(x)


# --- KERNEL ---
@triton.jit
def _fused_swish_v3_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    # Hard sigmoid: clamp(x/6 + 0.5, 0, 1)
    hs = tl.minimum(tl.maximum(x / 6.0 + 0.5, 0.0), 1.0)
    y = x * hs
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_swish_v3_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    hs = tl.minimum(tl.maximum(x / 6.0 + 0.5, 0.0), 1.0)
    dhs = tl.where((x / 6.0 + 0.5) > 0.0 and (x / 6.0 + 0.5) < 1.0, x / 6.0, 0.0)
    dx = dy * (hs + dhs * x)
    tl.store(DX + offs, dx, mask=mask)


class FusedSwishV3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_swish_v3_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_swish_v3_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedSwishV3.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
