"""Fused Sin activation kernel.

Pattern: sin(x)
Fuses elementwise sine.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return torch.sin(x)


# --- KERNEL ---
@triton.jit
def _fused_sin_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.sin(x)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_sin_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    dx = dy * tl.cos(x)
    tl.store(DX + offs, dx, mask=mask)


class FusedSin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_c = x.contiguous()
        y = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_sin_fwd[(n_blocks,)](x_c, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x_c)
        return y

    @staticmethod
    def backward(ctx, dy):
        x_c, = ctx.saved_tensors
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_sin_bwd[(n_blocks,)](x_c, dy_c, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedSin.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
