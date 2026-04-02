"""Fused Square-Add kernel.

Pattern: x^2 + y
Fuses square then add.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, y):
    return x * x + y


# --- KERNEL ---
@triton.jit
def _fused_square_add_fwd(X, Y, Z, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Z + offs, x * x + y, mask=mask)


@triton.jit
def _fused_square_add_bwd(X, Y, DZ, DX, DY, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dz = tl.load(DZ + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(DX + offs, 2.0 * x * dz, mask=mask)
    tl.store(DY + offs, dz, mask=mask)


class FusedSquareAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        x_c = x.contiguous()
        y_c = y.contiguous()
        z = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_square_add_fwd[(n_blocks,)](x_c, y_c, z, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x_c, y_c)
        return z

    @staticmethod
    def backward(ctx, dz):
        x_c, y_c = ctx.saved_tensors
        dz_c = dz.contiguous()
        dx = torch.empty_like(x_c)
        dy = torch.empty_like(y_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_square_add_bwd[(n_blocks,)](x_c, y_c, dz_c, dx, dy, N, BLOCK=BLOCK, num_warps=4)
        return dx, dy


def kernel_fn(x, y):
    return FusedSquareAdd.apply(x, y)


def can_use_kernel(x, y):
    return (x.is_cuda and y.is_cuda and
            x.is_contiguous() and y.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32) and
            x.shape == y.shape)


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
