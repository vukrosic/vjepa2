"""Fused Channel Shift kernel.

Pattern: y[c] = x[(c + shift_offset) % C]
Fuses: channel permutation/shift operation into one contiguous memory pass.
Like Deformable Conv channel shuffle, used in some V-JEPA variant attention.
Uses pure scalar loads over channel dimension.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, shift):
    B, H, C = x.shape
    y = torch.empty_like(x)
    for c in range(C):
        src_c = (c + shift) % C
        y[..., c] = x[..., src_c]
    return y


# --- KERNEL ---
@triton.jit
def _fused_channel_shift_fwd(
    X, Y,
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
    shift: tl.constexpr,
    stride_xb, stride_xh, stride_xc,
    stride_yb, stride_yh, stride_yc,
    BLOCK_C: tl.constexpr,
):
    """One program per (b, h). Pure scalar loads over C."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_base = b * stride_xb + h * stride_xh
    y_base = b * stride_yb + h * stride_yh

    for c in range(C):
        src_c = (c + shift) % C
        x_val = tl.load(X + row_base + src_c * stride_xc).to(tl.float32)
        tl.store(Y + y_base + c * stride_yc, x_val)


@triton.jit
def _fused_channel_shift_bwd(
    X, Y, DX, DY,
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
    shift: tl.constexpr,
    stride_xb, stride_xh, stride_xc,
    stride_dxb, stride_dxh, stride_dxc,
    stride_dyb, stride_dyh, stride_dyc,
    BLOCK_C: tl.constexpr,
):
    """Backward: reverse shift."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_base = b * stride_xb + h * stride_xh
    dy_base = b * stride_dyb + h * stride_dyh
    dx_base = b * stride_dxb + h * stride_dxh

    # Reverse shift for backward: src[c] = dst[(c + shift) % C]
    for c in range(C):
        src_c = (c + shift) % C
        dy_val = tl.load(DY + dy_base + src_c * stride_dyc).to(tl.float32)
        tl.store(DX + dx_base + c * stride_dxc, dy_val)


class FusedChannelShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.shift = shift
        B, H, C = x.shape
        y = torch.empty_like(x)
        grid = (B, H)
        _fused_channel_shift_fwd[grid](
            x, y,
            B=B, H=H, C=C,
            shift=shift,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_C=triton.next_power_of_2(C),
            num_warps=min(16, max(1, C // 32)),
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        shift = ctx.shift
        B, H, C = x.shape
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        grid = (B, H)
        _fused_channel_shift_bwd[grid](
            x, dy, dx, dy,
            B=B, H=H, C=C,
            shift=shift,
            x.stride(0), x.stride(1), x.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            dy.stride(0), dy.stride(1), dy.stride(2),
            BLOCK_C=triton.next_power_of_2(C),
            num_warps=min(16, max(1, C // 32)),
        )
        return dx, None


def kernel_fn(x, shift):
    return FusedChannelShift.apply(x, shift)


def can_use_kernel(x, shift):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            0 <= shift < x.shape[-1])


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "shift": 128},
    "vit_h": {"x": (2, 2048, 1280), "shift": 160},
    "small": {"x": (4, 256, 384), "shift": 48},
}
