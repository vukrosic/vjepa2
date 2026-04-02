"""Fused Subtract Mean + Scale kernel.

Pattern: y = (x - mean_per_channel) * gamma
Fuses: per-channel mean subtraction and gamma scaling into one pass.
Common in GroupNorm normalization stages.
Uses pure scalar loads per channel.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, mean, gamma):
    # mean: (C,) gamma: (C,)
    # Subtract per-channel mean then scale
    return (x - mean.view(1, 1, -1)) * gamma.view(1, 1, -1)


# --- KERNEL ---
@triton.jit
def _fused_sub_mean_scale_fwd(
    X, Mean, Gamma, Y,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
    stride_xb, stride_xh, stride_xc,
    stride_yb, stride_yh, stride_yc,
    BLOCK_C: tl.constexpr,
):
    """One program per (b, h) position. Pure scalar loads over C."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_base = b * stride_xb + h * stride_xh
    y_base = b * stride_yb + h * stride_yh

    # Pure scalar load over C
    for c in range(C):
        x_val = tl.load(X + row_base + c * stride_xc).to(tl.float32)
        mean_val = tl.load(Mean + c).to(tl.float32)
        gamma_val = tl.load(Gamma + c).to(tl.float32)
        y_val = (x_val - mean_val) * gamma_val
        tl.store(Y + y_base + c * stride_yc, y_val)


@triton.jit
def _fused_sub_mean_scale_bwd(
    X, Mean, Gamma, DY, DX, DGamma,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
    stride_xb, stride_xh, stride_xc,
    stride_dyb, stride_dyh, stride_dyc,
    BLOCK_C: tl.constexpr,
):
    """Backward: dX = dy * gamma, dGamma = sum(dy * (x - mean))."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_base = b * stride_xb + h * stride_xh
    dy_base = b * stride_dyb + h * stride_dyh

    # Accumulate dGamma per channel using atomic add
    for c in range(C):
        x_val = tl.load(X + row_base + c * stride_xc).to(tl.float32)
        mean_val = tl.load(Mean + c).to(tl.float32)
        gamma_val = tl.load(Gamma + c).to(tl.float32)
        dy_val = tl.load(DY + dy_base + c * stride_dyc).to(tl.float32)
        dx_val = dy_val * gamma_val
        tl.store(DX + row_base + c * stride_xc, dx_val)
        # dGamma = dy * (x - mean), accumulated across H
        dgamma_val = dy_val * (x_val - mean_val)
        tl.atomic_add(DGamma + c, dgamma_val)


class FusedSubtractMeanScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mean, gamma):
        assert x.is_contiguous() and mean.is_contiguous() and gamma.is_contiguous()
        ctx.save_for_backward(x, mean, gamma)
        B, H, C = x.shape
        y = torch.empty_like(x)
        grid = (B, H)
        _fused_sub_mean_scale_fwd[grid](
            x, mean, gamma, y,
            B=B, C=C, H=H,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_C=triton.next_power_of_2(C),
            num_warps=min(16, max(1, C // 32)),
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        x, mean, gamma = ctx.saved_tensors
        B, H, C = x.shape
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        dgamma = torch.zeros(C, dtype=torch.float32, device=x.device)
        grid = (B, H)
        _fused_sub_mean_scale_bwd[grid](
            x, mean, gamma, dy, dx, dgamma,
            B=B, C=C, H=H,
            x.stride(0), x.stride(1), x.stride(2),
            dy.stride(0), dy.stride(1), dy.stride(2),
            BLOCK_C=triton.next_power_of_2(C),
            num_warps=min(16, max(1, C // 32)),
        )
        # dMean = -sum(dy * gamma) across B*H
        dmean = -dgamma.view(B, H, C).sum(dim=(0, 1))
        return dx, dmean.to(mean.dtype), dgamma.to(gamma.dtype)


def kernel_fn(x, mean, gamma):
    return FusedSubtractMeanScale.apply(x, mean, gamma)


def can_use_kernel(x, mean, gamma):
    return (x.is_cuda and mean.is_cuda and gamma.is_cuda and
            x.is_contiguous() and mean.is_contiguous() and gamma.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            x.shape[-1] == mean.shape[0] == gamma.shape[0])


SHAPES = {
    "vit_l": {"x": (2, 16, 1024), "C": 1024},
    "vit_h": {"x": (2, 16, 1280), "C": 1280},
    "small": {"x": (4, 8, 384), "C": 384},
}
