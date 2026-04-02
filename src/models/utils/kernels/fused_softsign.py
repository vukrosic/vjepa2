"""Fused Softsign activation kernel.

Pattern: x / (1 + |x|)
Fuses elementwise softsign.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return torch.nn.functional.softsign(x)


# --- KERNEL ---
@triton.jit
def _fused_softsign_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = x / (1.0 + tl.abs(x))
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_softsign_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    ax = 1.0 + tl.abs(x)
    dx = dy / (ax * ax)
    tl.store(DX + offs, dx, mask=mask)


class FusedSoftsign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_c = x.contiguous()
        y = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_softsign_fwd[(n_blocks,)](x_c, y, N, BLOCK=BLOCK, num_warps=4)
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
        _fused_softsign_bwd[(n_blocks,)](x_c, dy_c, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedSoftsign.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
