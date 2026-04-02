"""Fused Exact GELU activation kernel.

Pattern: y = x * 0.5 * (1 + erf(x / sqrt(2)))
Exact GELU using the error function (not the approximate form).
Fuses: erf + multiply + add into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return torch.nn.functional.gelu(x, approximate="none")


# --- KERNEL ---
@triton.jit
def _fused_gelu_exact_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    k = 0.7071067811865476  # 1/sqrt(2)
    y = 0.5 * x * (1.0 + tl.erf(x * k))
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_gelu_exact_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    # d/dx GELU = 0.5 * (1 + erf(x/sqrt(2))) + 0.5 * x * d(erf)/dx
    # d(erf)/dx = 2/sqrt(pi) * exp(-x^2/2)
    k = 0.7071067811865476
    dgelu = 0.5 * (1.0 + tl.erf(x * k)) + 0.5 * x * (2.0 / 1.772453850905516) * tl.exp(-0.5 * x * x)
    dx = dy * dgelu
    tl.store(DX + offs, dx, mask=mask)


class FusedGELUExact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_exact_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_exact_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedGELUExact.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float32, torch.float16, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
