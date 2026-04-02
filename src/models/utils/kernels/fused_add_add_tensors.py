"""Fused Add-Add Tensors kernel.

Pattern: (x + y) + z = x + y + z
Fuses three-tensor chained addition.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, y, z):
    return x + y + z


# --- KERNEL ---
@triton.jit
def _fused_add_add_fwd(X, Y, Z, W, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(W + offs, x + y + z, mask=mask)


@triton.jit
def _fused_add_add_bwd(X, Y, Z, DW, DX, DY, DZ, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    dw = tl.load(DW + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(DX + offs, dw, mask=mask)
    tl.store(DY + offs, dw, mask=mask)
    tl.store(DZ + offs, dw, mask=mask)


class FusedAddAddTensors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, z):
        x_c = x.contiguous()
        y_c = y.contiguous()
        z_c = z.contiguous()
        w = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_add_add_fwd[(n_blocks,)](x_c, y_c, z_c, w, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x_c, y_c, z_c)
        return w

    @staticmethod
    def backward(ctx, dw):
        x_c, y_c, z_c = ctx.saved_tensors
        dw_c = dw.contiguous()
        dx = torch.empty_like(x_c)
        dy = torch.empty_like(y_c)
        dz = torch.empty_like(z_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_add_add_bwd[(n_blocks,)](x_c, y_c, z_c, dw_c, dx, dy, dz, N, BLOCK=BLOCK, num_warps=4)
        return dx, dy, dz


def kernel_fn(x, y, z):
    return FusedAddAddTensors.apply(x, y, z)


def can_use_kernel(x, y, z):
    return (x.is_cuda and y.is_cuda and z.is_cuda and
            x.is_contiguous() and y.is_contiguous() and z.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32) and
            x.shape == y.shape == z.shape)


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
