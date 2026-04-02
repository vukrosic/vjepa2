"""Fused SoftPlus activation kernel.

Pattern: y = log(1 + exp(x)) / beta scaled by threshold
Fuses: exp + log + add into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, beta=1.0, threshold=20.0):
    return torch.nn.functional.softplus(x, beta=beta, threshold=threshold)


# --- KERNEL ---
@triton.jit
def _fused_soft_plus_fwd(X, Y, N: tl.constexpr, BETA: tl.constexpr, THRESH: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.where(x * BETA < THRESH, tl.log(1.0 + tl.exp(tl.minimum(x * BETA, THRESH))) / BETA, x)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_soft_plus_bwd(X, DY, DX, N: tl.constexpr, BETA: tl.constexpr, THRESH: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-BETA * x))
    dx = dy * tl.where(x * BETA < THRESH, sig, 1.0)
    tl.store(DX + offs, dx, mask=mask)


class FusedSoftPlus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta=1.0, threshold=20.0):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.beta = beta
        ctx.threshold = threshold
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_soft_plus_fwd[grid](x, y, N, beta, threshold, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_soft_plus_bwd[grid](x, dy, dx, N, ctx.beta, ctx.threshold, BLOCK=BLOCK, num_warps=4)
        return dx, None, None


def kernel_fn(x, beta=1.0, threshold=20.0):
    return FusedSoftPlus.apply(x, beta, threshold)


def can_use_kernel(x, beta=1.0, threshold=20.0):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
