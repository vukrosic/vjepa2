"""Fused Swish/SiLU activation kernel.

Pattern: x * sigmoid(x) = swish(x) = silu(x)
Fuses: sigmoid + multiply into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x):
    return torch.nn.functional.silu(x)


# --- KERNEL ---
@triton.jit
def _fused_swish_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    # Swish: x * sigmoid(x) = x / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-x, 20.0)))
    y = x * sig
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_swish_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    # Forward: y = x * sigmoid(x)
    # d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-x, 20.0)))
    dsig_dx = sig * (1.0 - sig)
    # d(out)/dx = sigmoid(x) + x * dsig_dx = sig + x * sig * (1 - sig)
    dx = dy * (sig + x * dsig_dx)
    tl.store(DX + offs, dx, mask=mask)


class FusedSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_swish_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_swish_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedSwish.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
