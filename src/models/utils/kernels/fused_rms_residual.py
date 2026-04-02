"""Fused RMSNorm + Residual Add kernel.

Pattern: y = x + residual; y_norm = y / rms(y) * weight
Fuses: residual add + RMSNorm (no mean centering) into one pass.
RMSNorm = x / sqrt(mean(x^2) + eps) * weight, skips mean subtraction.
Uses pure scalar loads over normalized dimension.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, residual, weight, eps=1e-6):
    # Add residual then RMSNorm
    y = x + residual
    rms = y.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return (y / rms * weight).to(x.dtype)


# --- KERNEL ---
@triton.jit
def _fused_rms_residual_fwd(
    X, R, W, Y,
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
    stride_xb, stride_xh, stride_xc,
    stride_rxb, stride_rxh, stride_rxc,
    stride_yb, stride_yh, stride_yc,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One program per (b, h). Pure scalar loads over C."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_x = b * stride_xb + h * stride_xh
    row_r = b * stride_rxb + h * stride_rxh
    row_y = b * stride_yb + h * stride_yh

    # Compute sum of squares of (x + r)
    ss = 0.0
    for c in range(C):
        x_val = tl.load(X + row_x + c * stride_xc).to(tl.float32)
        r_val = tl.load(R + row_r + c * stride_rxc).to(tl.float32)
        y_val = x_val + r_val
        ss += y_val * y_val

    rms_inv = 1.0 / tl.sqrt(ss / C + eps)

    # Normalize and apply weight
    for c in range(C):
        x_val = tl.load(X + row_x + c * stride_xc).to(tl.float32)
        r_val = tl.load(R + row_r + c * stride_rxc).to(tl.float32)
        y_val = x_val + r_val
        w_val = tl.load(W + c).to(tl.float32)
        out = y_val * rms_inv * w_val
        tl.store(Y + row_y + c * stride_yc, out)


@triton.jit
def _fused_rms_residual_bwd(
    X, R, W, DY, DX, DR, DW,
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
    stride_xb, stride_xh, stride_xc,
    stride_rxb, stride_rxh, stride_rxc,
    stride_dyb, stride_dyh, stride_dyc,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Backward: computes dX, dResidual, dWeight."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_x = b * stride_xb + h * stride_xh
    row_r = b * stride_rxb + h * stride_rxh
    row_dy = b * stride_dyb + h * stride_dyh

    # Recompute sum of squares
    ss = 0.0
    for c in range(C):
        x_val = tl.load(X + row_x + c * stride_xc).to(tl.float32)
        r_val = tl.load(R + row_r + c * stride_rxc).to(tl.float32)
        y_val = x_val + r_val
        ss += y_val * y_val
    rms = tl.sqrt(ss / C + eps)
    rms_inv = 1.0 / rms

    # Compute sum(dy * w * y) for normalization correction
    dot = 0.0
    for c in range(C):
        x_val = tl.load(X + row_x + c * stride_xc).to(tl.float32)
        r_val = tl.load(R + row_r + c * stride_rxc).to(tl.float32)
        y_val = x_val + r_val
        w_val = tl.load(W + c).to(tl.float32)
        dy_val = tl.load(DY + row_dy + c * stride_dyc).to(tl.float32)
        dot += dy_val * w_val * y_val

    rms3 = rms * rms * rms  # rms^3
    correction = dot / (C * rms3)

    for c in range(C):
        x_val = tl.load(X + row_x + c * stride_xc).to(tl.float32)
        r_val = tl.load(R + row_r + c * stride_rxc).to(tl.float32)
        y_val = x_val + r_val
        w_val = tl.load(W + c).to(tl.float32)
        dy_val = tl.load(DY + row_dy + c * stride_dyc).to(tl.float32)
        # dx = (dy * w - y * dot / (N * rms^2)) / rms
        dx = (dy_val * w_val - y_val * correction) * rms_inv
        dr = dx  # same gradient for residual
        tl.store(DX + row_x + c * stride_xc, dx)
        tl.store(DR + row_r + c * stride_rxc, dr)
        # dw = dy * y * rms_inv
        dw = dy_val * y_val * rms_inv
        tl.atomic_add(DW + c, dw)


class FusedRMSResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, eps=1e-6):
        assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
        ctx.save_for_backward(x, residual, weight)
        ctx.eps = eps
        B, H, C = x.shape
        y = torch.empty_like(x)
        grid = (B, H)
        _fused_rms_residual_fwd[grid](
            x, residual, weight, y,
            B=B, H=H, C=C,
            x.stride(0), x.stride(1), x.stride(2),
            residual.stride(0), residual.stride(1), residual.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            eps=eps,
            BLOCK_C=triton.next_power_of_2(C),
            num_warps=min(16, max(1, C // 32)),
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        x, residual, weight = ctx.saved_tensors
        eps = ctx.eps
        B, H, C = x.shape
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        dr = torch.empty_like(residual)
        dw = torch.zeros(C, dtype=torch.float32, device=x.device)
        grid = (B, H)
        _fused_rms_residual_bwd[grid](
            x, residual, weight, dy, dx, dr, dw,
            B=B, H=H, C=C,
            x.stride(0), x.stride(1), x.stride(2),
            residual.stride(0), residual.stride(1), residual.stride(2),
            dy.stride(0), dy.stride(1), dy.stride(2),
            eps=eps,
            BLOCK_C=triton.next_power_of_2(C),
            num_warps=min(16, max(1, C // 32)),
        )
        return dx, dr, dw.to(weight.dtype), None


def kernel_fn(x, residual, weight, eps=1e-6):
    return FusedRMSResidual.apply(x, residual, weight, eps)


def can_use_kernel(x, residual, weight):
    return (x.is_cuda and residual.is_cuda and weight.is_cuda and
            x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous() and
            x.shape == residual.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            x.shape[-1] == weight.shape[0])


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "C": 1024},
    "vit_h": {"x": (2, 2048, 1280), "C": 1280},
    "small": {"x": (4, 256, 384), "C": 384},
}
