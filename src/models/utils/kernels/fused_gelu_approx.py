"""Fused GELU approximation kernel using tanh.

Pattern: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
Fuses: polynomial + tanh + multiply into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x):
    return torch.nn.functional.gelu(x)


# --- KERNEL ---
@triton.jit
def _fused_gelu_approx_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    # GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Compute x^3 without power operator: x * x * x
    x_sq = x * x
    x_cu = x_sq * x
    # sqrt(2/pi) ≈ 0.79788456
    alpha = 0.79788456
    beta = 0.044715
    u = alpha * (x + beta * x_cu)
    # tanh(u) = (exp(2u) - 1) / (exp(2u) + 1)
    e2u = tl.exp(tl.minimum(2.0 * u, 40.0))
    tanh_u = (e2u - 1.0) / (e2u + 1.0)
    y = 0.5 * x * (1.0 + tanh_u)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_gelu_approx_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    # GELU derivative: 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh^2(u)) * du/dx
    # where u = alpha * (x + beta * x^3), du/dx = alpha * (1 + 3*beta*x^2)
    x_sq = x * x
    x_cu = x_sq * x
    alpha = 0.79788456
    beta = 0.044715
    u = alpha * (x + beta * x_cu)
    e2u = tl.exp(tl.minimum(2.0 * u, 40.0))
    tanh_u = (e2u - 1.0) / (e2u + 1.0)
    dtanh_du = 1.0 - tanh_u * tanh_u
    du_dx = alpha * (1.0 + 3.0 * beta * x_sq)
    dgelu_dx = 0.5 * (1.0 + tanh_u) + 0.5 * x * dtanh_du * du_dx
    dx = dy * dgelu_dx
    tl.store(DX + offs, dx, mask=mask)


class FusedGeluApprox(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_approx_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_approx_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedGeluApprox.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
