"""Fused Divide Tensors kernel.

Pattern: y = x / (other + eps)
Fuses: add eps + divide into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, other, eps=1e-5):
    return x / (other + eps)


# --- KERNEL ---
@triton.jit
def _fused_div_tensor_fwd(X, Other, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    o = tl.load(Other + offs, mask=mask).to(tl.float32)
    y = x / (o + EPS)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_div_tensor_bwd(X, Other, DY, DX, DO, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    o = tl.load(Other + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    denom = o + EPS
    tl.store(DX + offs, dy / denom, mask=mask)
    tl.store(DO + offs, -dy * x / (denom * denom), mask=mask)


class FusedDivTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, other, eps=1e-5):
        assert x.is_contiguous() and other.is_contiguous()
        ctx.save_for_backward(x, other)
        ctx.eps = eps
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_div_tensor_fwd[grid](x, other, y, N, eps, BLOCK=BLOCK, num_warps=4)
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
        _fused_div_tensor_bwd[grid](x, other, dy, dx, do, N, ctx.eps, BLOCK=BLOCK, num_warps=4)
        return dx, do, None


def kernel_fn(x, other, eps=1e-5):
    return FusedDivTensor.apply(x, other, eps)


def can_use_kernel(x, other, eps=1e-5):
    return (x.is_cuda and other.is_cuda and
            x.is_contiguous() and other.is_contiguous() and
            x.shape == other.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096), "other": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120), "other": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536), "other": (8, 256, 1536)},
}
