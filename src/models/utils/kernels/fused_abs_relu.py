"""Fused Abs + Relu kernel.

Pattern: y = relu(|x|) + relu(other) = |x| + relu(other) (since |x| >= 0 always)
Fuses: abs + relu + add into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, other):
    return torch.abs(x) + torch.nn.functional.relu(other)


# --- KERNEL ---
@triton.jit
def _fused_abs_relu_fwd(X, Other, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    o = tl.load(Other + offs, mask=mask).to(tl.float32)
    ax = tl.where(x >= 0, x, -x)
    ro = tl.where(o > 0, o, 0.0)
    y = ax + ro
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_abs_relu_bwd(X, Other, DY, DX, DO, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    o = tl.load(Other + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    dx = tl.where(x >= 0, dy, -dy)
    do = tl.where(o > 0, dy, 0.0)
    tl.store(DX + offs, dx, mask=mask)
    tl.store(DO + offs, do, mask=mask)


class FusedAbsRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, other):
        assert x.is_contiguous() and other.is_contiguous()
        ctx.save_for_backward(x, other)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_abs_relu_fwd[grid](x, other, y, N, BLOCK=BLOCK, num_warps=4)
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
        _fused_abs_relu_bwd[grid](x, other, dy, dx, do, N, BLOCK=BLOCK, num_warps=4)
        return dx, do


def kernel_fn(x, other):
    return FusedAbsRelu.apply(x, other)


def can_use_kernel(x, other):
    return (x.is_cuda and other.is_cuda and
            x.is_contiguous() and other.is_contiguous() and
            x.shape == other.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096), "other": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120), "other": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536), "other": (8, 256, 1536)},
}
