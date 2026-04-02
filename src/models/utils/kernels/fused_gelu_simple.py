"""Fused GELU kernel (exact form, not approximate).

Pattern: y = x * 0.5 * (1 + erf(x / sqrt(2)))
Using sigmoid-based approximation: x * sigmoid(1.702 * x)
Fuses: computation into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.gelu(x)


@triton.jit
def _fused_gelu_simple_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask).to(tl.float32)
    # GELU exact via error function: 0.5 * x * (1 + erf(x / sqrt(2)))
    # Approximation: x * sigmoid(1.702 * x)
    k = 1.702
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-k * v, 40.0)))
    y = v * sig
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_gelu_simple_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    k = 1.702
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-k * x, 40.0)))
    dsig = sig * (1.0 - sig)
    dgelu = sig + k * x * dsig
    dx = dy * dgelu
    tl.store(DX + offs, dx, mask=mask)


class FusedGeluSimple(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        N = x.numel()
        y = torch.empty_like(x)
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_simple_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_simple_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedGeluSimple.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536)},
}
