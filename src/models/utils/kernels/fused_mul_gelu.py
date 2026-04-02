"""Fused Mul + GELU kernel.

Pattern: y = gelu(x) * other — fused multiply with GELU activation.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, other):
    return torch.nn.functional.gelu(x) * other


@triton.jit
def _gelu_approx(v):
    c1 = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    return 0.5 * v * (1.0 + tl.libdevice.tanh(sqrt_2_over_pi * (v + c1 * v * v * v)))


@triton.jit
def _mul_gelu_fwd(X, Other, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    ov = tl.load(Other + offs, mask=mask).to(tl.float32)
    gv = _gelu_approx(xv)
    tl.store(Y + offs, gv * ov, mask=mask)


@triton.jit
def _mul_gelu_bwd(X, Other, Y, DY, DX, DO, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    ov = tl.load(Other + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    c1 = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    tanh_arg = sqrt_2_over_pi * (xv + c1 * xv * xv * xv)
    tanh_v = tl.libdevice.tanh(tanh_arg)
    dtanh = 1.0 - tanh_v * tanh_v
    dgelu_dx = 0.5 * (1.0 + tanh_v) + 0.5 * xv * dtanh * sqrt_2_over_pi * (1.0 + 3.0 * c1 * xv * xv)
    tl.store(DX + offs, dy * ov * dgelu_dx, mask=mask)
    gv = 0.5 * xv * (1.0 + tanh_v)
    tl.store(DO + offs, dy * gv, mask=mask)


class FusedMulGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, other):
        assert x.is_contiguous() and other.is_contiguous()
        ctx.save_for_backward(x, other)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _mul_gelu_fwd[grid](x, other, y, N, BLOCK=BLOCK, num_warps=4)
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
        _mul_gelu_bwd[grid](x, other, ctx.needs_input_grad[0], dy, dx, do, N, BLOCK=BLOCK, num_warps=4)
        return dx, do


def kernel_fn(x, other):
    return FusedMulGELU.apply(x, other)


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
