"""Fused multiply by scalar kernel.

Pattern: y = x * scalar
Fuses: scalar multiply into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, scalar):
    return x * scalar


# --- KERNEL ---
@triton.jit
def _fused_mul_scalar_fwd(X, Y, N: tl.constexpr, SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = x * SCALAR
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_mul_scalar_bwd(X, DY, DX, N: tl.constexpr, SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    dx = dy * SCALAR
    tl.store(DX + offs, dx, mask=mask)


class FusedMulScalar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scalar):
        assert x.is_contiguous()
        ctx.save_for_backward(torch.tensor([scalar], device=x.device, dtype=x.dtype))
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_mul_scalar_fwd[grid](x, y, N, scalar, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (scalar_tensor,) = ctx.saved_tensors
        scalar = scalar_tensor.item()
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        N = dy.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_mul_scalar_bwd[grid](dy, dy, dx, N, scalar, BLOCK=BLOCK, num_warps=4)
        return dx, None


def kernel_fn(x, scalar):
    return FusedMulScalar.apply(x, scalar)


def can_use_kernel(x, scalar):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            isinstance(scalar, (int, float)))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "scalar": 0.1},
    "vit_h":  {"x": (2, 2048, 5120), "scalar": 2.0},
    "small":  {"x": (8, 256, 1536), "scalar": 0.5},
}
