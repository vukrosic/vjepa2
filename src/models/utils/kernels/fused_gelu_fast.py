"""Fused Fast GELU activation kernel.

Pattern: y = x * (1 + tanh(0.797885 * x + 0.035677 * x^3)) / 2
Alternative fast GELU approximation.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    # Approximate fast GELU
    k0 = 0.7978845608028654
    k1 = 0.044715
    return 0.5 * x * (1.0 + torch.tanh(k0 * x + k1 * x * x * x))


# --- KERNEL ---
@triton.jit
def _fused_gelu_fast_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    k0 = 0.7978845608028654
    k1 = 0.044715
    tanh_arg = k0 * x + k1 * x * x * x
    y = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_gelu_fast_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    k0 = 0.7978845608028654
    k1 = 0.044715
    tanh_arg = k0 * x + k1 * x * x * x
    tanh_val = tl.tanh(tanh_arg)
    d_tanh = 1.0 - tanh_val * tanh_val
    dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * x * d_tanh * (k0 + 3.0 * k1 * x * x)
    dx = dy * dgelu
    tl.store(DX + offs, dx, mask=mask)


class FusedGELUFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_fast_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_fast_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedGELUFast.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
