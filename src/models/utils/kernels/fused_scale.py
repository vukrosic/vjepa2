"""Fused Scale kernel.

Pattern: x * scale
Fuses elementwise scaling.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, scale):
    return x * scale


# --- KERNEL ---
@triton.jit
def _fused_scale_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, scale: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * scale
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_scale_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr, scale: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    dx = dy * scale
    tl.store(DX + offs, dx, mask=mask)


class FusedScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        x_c = x.contiguous()
        y = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_scale_fwd[(n_blocks,)](x_c, y, N, BLOCK=BLOCK, scale=scale, num_warps=4)
        ctx.save_for_backward(x_c)
        ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, dy):
        x_c, = ctx.saved_tensors
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_scale_bwd[(n_blocks,)](x_c, dy_c, dx, N, BLOCK=BLOCK, scale=ctx.scale, num_warps=4)
        return dx, None


def kernel_fn(x, scale):
    return FusedScale.apply(x, scale)


def can_use_kernel(x, scale):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
