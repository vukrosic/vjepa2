"""Fused Mul + Tanh kernel.

Pattern: y = tanh(x) * other — fused tanh activation times other tensor.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, other):
    return torch.tanh(x) * other


@triton.jit
def _mul_tanh_fwd(X, Other, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    ov = tl.load(Other + offs, mask=mask).to(tl.float32)
    e2x = tl.exp(tl.minimum(2.0 * xv, 40.0))
    tanh_v = (e2x - 1.0) / (e2x + 1.0)
    tl.store(Y + offs, tanh_v * ov, mask=mask)


@triton.jit
def _mul_tanh_bwd(X, Other, Y, DY, DX, DO, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    ov = tl.load(Other + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    e2x = tl.exp(tl.minimum(2.0 * xv, 40.0))
    tanh_v = (e2x - 1.0) / (e2x + 1.0)
    dtanh = 1.0 - tanh_v * tanh_v
    # d(tanh*other)/dx = dtanh * other
    tl.store(DX + offs, dy * ov * dtanh, mask=mask)
    # d(tanh*other)/dother = tanh
    tl.store(DO + offs, dy * tanh_v, mask=mask)


class FusedMulTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, other):
        assert x.is_contiguous() and other.is_contiguous()
        ctx.save_for_backward(x, other)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _mul_tanh_fwd[grid](x, other, y, N, BLOCK=BLOCK, num_warps=4)
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
        _mul_tanh_bwd[grid](x, other, ctx.needs_input_grad[0], dy, dx, do, N, BLOCK=BLOCK, num_warps=4)
        return dx, do


def kernel_fn(x, other):
    return FusedMulTanh.apply(x, other)


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
