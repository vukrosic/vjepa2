"""Fused Pow-Tensor kernel.

Pattern: y = x ** exponent
Fuses: power operation into one elementwise pass.
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, exponent):
    return x ** exponent


# --- KERNEL ---
@triton.jit
def _pow_tensor_fwd(X, Y, N: tl.constexpr, EXP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        v = tl.load(X + offs).to(tl.float32)
        y = tl.pow(v, EXP)
        tl.store(Y + offs, y)


@triton.jit
def _pow_tensor_bwd(X, Y, DY, DX, N: tl.constexpr, EXP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        x = tl.load(X + offs).to(tl.float32)
        dy = tl.load(DY + offs).to(tl.float32)
        # d/dx x^exp = exp * x^(exp-1)
        dx = dy * EXP * tl.pow(x, EXP - 1.0)
        tl.store(DX + offs, dx)


class FusedPowTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exponent):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.exponent = float(exponent)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _pow_tensor_fwd[grid](x, y, N, float(exponent), BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _pow_tensor_bwd[grid](x, x, dy, dx, N, ctx.exponent, BLOCK=BLOCK, num_warps=4)
        return dx, None


def kernel_fn(x, exponent):
    return FusedPowTensor.apply(x, exponent)


def can_use_kernel(x, exponent):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
