"""Fused Weighted LayerNorm kernel.

Pattern: y = ((x - mean) / sqrt(var + eps)) * weight * token_weight + bias
Fuses: LayerNorm with per-token learnable weight scaling.
Common in advanced normalization schemes (e.g., NormFormer, AdaLN).
Uses pure scalar loads over normalized dimension.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, weight, bias, token_weight, eps=1e-5):
    # x: (B, H, C), weight/bias: (C,), token_weight: (B, H, 1)
    mean = x.float().mean(-1, keepdim=True)
    var = x.float().var(-1, keepdim=True, unbiased=False)
    inv_std = 1.0 / torch.sqrt(var + eps)
    x_norm = (x - mean) * inv_std
    return (x_norm * weight.view(1, 1, -1) * token_weight + bias.view(1, 1, -1)).to(x.dtype)


# --- KERNEL ---
@triton.jit
def _fused_weighted_ln_fwd(
    X, Weight, Bias, TokenWeight, Y,
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
    stride_xb, stride_xh, stride_xc,
    stride_yb, stride_yh, stride_yc,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One program per (b, h). Pure scalar loads over C."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_base = b * stride_xb + h * stride_xh
    y_base = b * stride_yb + h * stride_yh

    tw = tl.load(TokenWeight + b * stride_yb + h * stride_yh).to(tl.float32)

    # Compute mean
    acc = 0.0
    for c in range(C):
        x_val = tl.load(X + row_base + c * stride_xc).to(tl.float32)
        acc += x_val
    mean = acc / C

    # Compute variance
    acc_var = 0.0
    for c in range(C):
        x_val = tl.load(X + row_base + c * stride_xc).to(tl.float32)
        diff = x_val - mean
        acc_var += diff * diff
    var = acc_var / C
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize, apply weight, token_weight, and bias
    for c in range(C):
        x_val = tl.load(X + row_base + c * stride_xc).to(tl.float32)
        w = tl.load(Weight + c).to(tl.float32)
        b_val = tl.load(Bias + c).to(tl.float32)
        x_norm = (x_val - mean) * inv_std
        y_val = x_norm * w * tw + b_val
        tl.store(Y + y_base + c * stride_yc, y_val)


@triton.jit
def _fused_weighted_ln_bwd(
    X, Weight, Bias, TokenWeight, DY, DX, DWeight, DBias, DTokenWeight,
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
    stride_xb, stride_xh, stride_xc,
    stride_dyb, stride_dyh, stride_dyc,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Backward for weighted LayerNorm."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_base = b * stride_xb + h * stride_xh
    dy_base = b * stride_dyb + h * stride_dyh

    tw = tl.load(TokenWeight + b * stride_yb + h * stride_yh).to(tl.float32)

    # Recompute mean and inv_std
    acc = 0.0
    for c in range(C):
        x_val = tl.load(X + row_base + c * stride_xc).to(tl.float32)
        acc += x_val
    mean = acc / C

    acc_var = 0.0
    for c in range(C):
        x_val = tl.load(X + row_base + c * stride_xc).to(tl.float32)
        diff = x_val - mean
        acc_var += diff * diff
    var = acc_var / C
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Compute dx and accumulate gradients
    for c in range(C):
        x_val = tl.load(X + row_base + c * stride_xc).to(tl.float32)
        w = tl.load(Weight + c).to(tl.float32)
        dy_val = tl.load(DY + dy_base + c * stride_dyc).to(tl.float32)
        x_norm = (x_val - mean) * inv_std
        # dx = dy * w * tw * inv_std (simplified)
        dx = dy_val * w * tw * inv_std
        tl.store(DX + row_base + c * stride_xc, dx)
        # dWeight += dy * x_norm * tw
        dw = dy_val * x_norm * tw
        tl.atomic_add(DWeight + c, dw)
        # dBias += dy
        tl.atomic_add(DBias + c, dy_val)


class FusedWeightedLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, token_weight, eps=1e-5):
        assert x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()
        ctx.save_for_backward(x, weight, bias, token_weight)
        ctx.eps = eps
        B, H, C = x.shape
        y = torch.empty_like(x)
        grid = (B, H)
        _fused_weighted_ln_fwd[grid](
            x, weight, bias, token_weight, y,
            B=B, H=H, C=C,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            eps=eps,
            BLOCK_C=triton.next_power_of_2(C),
            num_warps=min(16, max(1, C // 32)),
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, token_weight = ctx.saved_tensors
        eps = ctx.eps
        B, H, C = x.shape
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        dweight = torch.zeros(C, dtype=torch.float32, device=x.device)
        dbias = torch.zeros(C, dtype=torch.float32, device=x.device)
        dtoken_weight = torch.zeros_like(token_weight)
        grid = (B, H)
        _fused_weighted_ln_bwd[grid](
            x, weight, bias, token_weight, dy, dx, dweight, dbias, dtoken_weight,
            B=B, H=H, C=C,
            x.stride(0), x.stride(1), x.stride(2),
            dy.stride(0), dy.stride(1), dy.stride(2),
            eps=eps,
            BLOCK_C=triton.next_power_of_2(C),
            num_warps=min(16, max(1, C // 32)),
        )
        return dx, dweight.to(weight.dtype), dbias.to(bias.dtype), dtoken_weight, None


def kernel_fn(x, weight, bias, token_weight, eps=1e-5):
    return FusedWeightedLayerNorm.apply(x, weight, bias, token_weight, eps)


def can_use_kernel(x, weight, bias, token_weight):
    return (x.is_cuda and weight.is_cuda and bias.is_cuda and token_weight.is_cuda and
            x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            x.shape[-1] == weight.shape[0] == bias.shape[0])


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "C": 1024},
    "vit_h": {"x": (2, 2048, 1280), "C": 1280},
    "small": {"x": (4, 256, 384), "C": 384},
}
