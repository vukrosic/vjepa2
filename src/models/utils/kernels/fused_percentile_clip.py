"""Fused Percentile Clip kernel.

Pattern: y = clamp(x, low_per_channel, high_per_channel)
Fuses: per-channel percentile computation + clip in one kernel.
Useful for outlier suppression in transformer activations.
Uses pure scalar loads over sequence dimension.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, low, high):
    # low, high: (C,) tensors
    return torch.clamp(x, low.view(1, 1, -1), high.view(1, 1, -1))


# --- KERNEL ---
@triton.jit
def _fused_pctile_clip_fwd(
    X, Low, High, Y,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
    stride_xb, stride_xh, stride_xc,
    stride_yb, stride_yh, stride_yc,
    BLOCK_H: tl.constexpr,
):
    """One program per (b, c). Pure scalar loads over H (sequence)."""
    b = tl.program_id(0)
    c = tl.program_id(1)
    row_base = b * stride_xb + c * stride_xc
    y_base = b * stride_yb + c * stride_yc

    low = tl.load(Low + c).to(tl.float32)
    high = tl.load(High + c).to(tl.float32)

    for h in range(H):
        x_val = tl.load(X + row_base + h * stride_xh).to(tl.float32)
        y_val = tl.minimum(tl.maximum(x_val, low), high)
        tl.store(Y + y_base + h * stride_yh, y_val)


class FusedPercentileClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, low, high):
        assert x.is_contiguous() and low.is_contiguous() and high.is_contiguous()
        ctx.save_for_backward(x, low, high)
        B, H, C = x.shape
        y = torch.empty_like(x)
        grid = (B, C)
        _fused_pctile_clip_fwd[grid](
            x, low, high, y,
            B=B, C=C, H=H,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_H=triton.next_power_of_2(H),
            num_warps=min(16, max(1, H // 32)),
        )
        return y


def kernel_fn(x, low, high):
    return FusedPercentileClip.apply(x, low, high)


def can_use_kernel(x, low, high):
    return (x.is_cuda and low.is_cuda and high.is_cuda and
            x.is_contiguous() and low.is_contiguous() and high.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            x.shape[-1] == low.shape[0] == high.shape[0])


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "C": 1024},
    "vit_h": {"x": (2, 2048, 1280), "C": 1280},
    "small": {"x": (4, 256, 384), "C": 384},
}
