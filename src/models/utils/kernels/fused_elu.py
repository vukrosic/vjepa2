"""Fused ELU activation kernel.

Pattern: x if x > 0 else alpha * (exp(x) - 1)
Fuses elementwise ELU.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, alpha=1.0):
    return torch.nn.functional.elu(x, alpha=alpha)


# --- KERNEL ---
@triton.jit
def _fused_elu_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, alpha: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(x > 0, x, alpha * (tl.exp(tl.minimum(x, 20.0)) - 1.0))
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_elu_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr, alpha: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    dx = tl.where(x > 0, dy, dy * alpha * tl.exp(tl.minimum(x, 20.0)))
    tl.store(DX + offs, dx, mask=mask)


class FusedELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        x_c = x.contiguous()
        y = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_elu_fwd[(n_blocks,)](x_c, y, N, BLOCK=BLOCK, alpha=alpha, num_warps=4)
        ctx.save_for_backward(x_c)
        ctx.alpha = alpha
        return y

    @staticmethod
    def backward(ctx, dy):
        x_c, = ctx.saved_tensors
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _fused_elu_bwd[(n_blocks,)](x_c, dy_c, dx, N, BLOCK=BLOCK, alpha=ctx.alpha, num_warps=4)
        return dx, None


def kernel_fn(x, alpha=1.0):
    return FusedELU.apply(x, alpha)


def can_use_kernel(x, alpha=1.0):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
