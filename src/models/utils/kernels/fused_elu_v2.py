"""Fused ELU v2 activation kernel.

Pattern: y = x if x >= 0 else alpha * (exp(x) - 1)
Fuses: exp + select + multiply into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, alpha=1.0):
    return torch.nn.functional.elu(x, alpha)


# --- KERNEL ---
@triton.jit
def _fused_elu_v2_fwd(X, Y, N: tl.constexpr, ALPHA: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.where(x >= 0, x, ALPHA * (tl.exp(tl.minimum(x, 40.0)) - 1.0))
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_elu_v2_bwd(X, DY, DX, N: tl.constexpr, ALPHA: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    dx = tl.where(x >= 0, dy, dy * ALPHA * tl.exp(tl.minimum(x, 40.0)))
    tl.store(DX + offs, dx, mask=mask)


class FusedELUV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_elu_v2_fwd[grid](x, y, N, alpha, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_elu_v2_bwd[grid](x, dy, dx, N, ctx.alpha, BLOCK=BLOCK, num_warps=4)
        return dx, None


def kernel_fn(x, alpha=1.0):
    return FusedELUV2.apply(x, alpha)


def can_use_kernel(x, alpha=1.0):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "alpha": 1.0},
    "vit_h":  {"x": (2, 2048, 5120), "alpha": 1.0},
    "small":  {"x": (8, 256, 1536), "alpha": 1.0},
}
