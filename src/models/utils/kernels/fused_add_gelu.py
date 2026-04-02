"""Fused Add + GELU kernel.

Pattern: y = gelu(x + other) — fused add + GELU activation.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, other):
    return torch.nn.functional.gelu(x + other)


@triton.jit
def _gelu_approx(v):
    # tanh approximation: gelu(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c1 = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    return 0.5 * v * (1.0 + tl.libdevice.tanh(sqrt_2_over_pi * (v + c1 * v * v * v)))


@triton.jit
def _add_gelu_fwd(X, Other, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    ov = tl.load(Other + offs, mask=mask).to(tl.float32)
    s = xv + ov
    yv = _gelu_approx(s)
    tl.store(Y + offs, yv, mask=mask)


@triton.jit
def _add_gelu_bwd(X, Other, Y, DY, DX, DO, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    ov = tl.load(Other + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    s = xv + ov
    c1 = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    # dGELU/ds = 0.5 * (1 + tanh(...)) + 0.5 * s * (1 - tanh^2(...)) * sqrt(2/pi) * (1 + 3*c1*s^2)
    tanh_arg = sqrt_2_over_pi * (s + c1 * s * s * s)
    tanh_v = tl.libdevice.tanh(tanh_arg)
    dtanh = 1.0 - tanh_v * tanh_v
    dgelu_ds = 0.5 * (1.0 + tanh_v) + 0.5 * s * dtanh * sqrt_2_over_pi * (1.0 + 3.0 * c1 * s * s)
    dx = dy * dgelu_ds
    tl.store(DX + offs, dx, mask=mask)
    tl.store(DO + offs, dx, mask=mask)


class FusedAddGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, other):
        assert x.is_contiguous() and other.is_contiguous()
        ctx.save_for_backward(x, other)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _add_gelu_fwd[grid](x, other, y, N, BLOCK=BLOCK, num_warps=4)
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
        _add_gelu_bwd[grid](x, other, ctx.needs_input_grad[0], dy, dx, do, N, BLOCK=BLOCK, num_warps=4)
        return dx, do


def kernel_fn(x, other):
    return FusedAddGELU.apply(x, other)


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
