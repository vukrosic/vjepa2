"""Fused Div Scale kernel.

Pattern: (x / y) * scale
Fuses divide then scale.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, y, scale):
    return (x / (y + 1e-12)) * scale


# --- KERNEL ---
@triton.jit
def _fused_div_scale_fwd(X, Y, Z, N: tl.constexpr, BLOCK: tl.constexpr, scale: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(Z + offs, x / (y + 1e-12) * scale, mask=mask)


@triton.jit
def _fused_div_scale_bwd(X, Y, DZ, DX, DY, N: tl.constexpr, BLOCK: tl.constexpr, scale: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + offs, mask=mask, other=1.0).to(tl.float32)
    dz = tl.load(DZ + offs, mask=mask, other=0.0).to(tl.float32)
    y_safe = y + 1e-12
    d_scaled = dz * scale
    tl.store(DX + offs, d_scaled / y_safe, mask=mask)
    tl.store(DY + offs, -d_scaled * x / (y_safe * y_safe), mask=mask)


class FusedDivScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, scale):
        x_c = x.contiguous()
        y_c = y.contiguous()
        z = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_div_scale_fwd[(n_blocks,)](x_c, y_c, z, N, BLOCK=BLOCK, scale=scale, num_warps=4)
        ctx.save_for_backward(x_c, y_c)
        ctx.scale = scale
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
        _fused_div_scale_bwd[(n_blocks,)](x_c, y_c, dz_c, dx, dy, N, BLOCK=BLOCK, scale=ctx.scale, num_warps=4)
        return dx, dy, None


def kernel_fn(x, y, scale):
    return FusedDivScale.apply(x, y, scale)


def can_use_kernel(x, y, scale):
    return (x.is_cuda and y.is_cuda and
            x.is_contiguous() and y.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32) and
            x.shape == y.shape)


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
