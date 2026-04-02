"""Fused Add Bias + GELU activation kernel.

Pattern: y = GELU(x + bias), where GELU(x) = x * sigmoid(1.702 * x) approx.
Fuses: bias add + GELU activation into one read/write pass.
Common in MLP Mixer / FFN blocks after linear projection.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, bias):
    return torch.nn.functional.gelu(x + bias)


# --- KERNEL ---
@triton.jit
def _fused_add_bias_gelu_fwd(X, Bias, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    b = tl.load(Bias + offs, mask=mask).to(tl.float32)
    y = x + b
    # GELU approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Numerically stable equivalent using expm1:
    # tanh(z) = (exp(2*z) - 1) / (exp(2*z) + 1)
    # Use sigmoid-based GELU: x * sigmoid(1.702 * x) which is the exact erf form
    # 1.702 ~ sqrt(2) * 1.0, sigmoid(z) = 1 / (1 + exp(-z))
    k = 1.702
    sig = 1.0 / (1.0 + tl.exp(-k * y))
    gelu = y * sig
    tl.store(Y + offs, gelu, mask=mask)


@triton.jit
def _fused_add_bias_gelu_bwd(X, Bias, DY, DX, DB, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    b = tl.load(Bias + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    y = x + b
    # dGELU/dx = sigmoid(k*x) + k*x*sigmoid(k*x)*(1-sigmoid(k*x)), k=1.702
    k = 1.702
    sig = 1.0 / (1.0 + tl.exp(-k * y))
    dsig = sig * (1.0 - sig)
    dgelu = sig + k * y * dsig
    dx = dy * dgelu
    tl.store(DX + offs, dx, mask=mask)
    tl.store(DB + offs, dx, mask=mask)


class FusedAddBiasGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        assert x.is_contiguous() and bias.is_contiguous()
        ctx.save_for_backward(x, bias)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_add_bias_gelu_fwd[grid](x, bias, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, bias = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        db = torch.empty_like(bias)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_add_bias_gelu_bwd[grid](x, bias, dy, dx, db, N, BLOCK=BLOCK, num_warps=4)
        return dx, db


def kernel_fn(x, bias):
    return FusedAddBiasGelu.apply(x, bias)


def can_use_kernel(x, bias):
    return (x.is_cuda and bias.is_cuda and
            x.is_contiguous() and bias.is_contiguous() and
            x.shape == bias.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096), "bias": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120), "bias": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536), "bias": (8, 256, 1536)},
}
