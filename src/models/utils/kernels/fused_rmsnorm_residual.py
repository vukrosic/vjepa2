"""Fused RMSNorm + Residual Add kernel.

Pattern: RMSNorm(x + residual) where RMSNorm = x / rms(x) * weight
Fuses: residual add + RMSNorm into one pass over the data.
Simpler than LayerNorm (no mean subtraction), should be faster.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, residual, weight, eps=1e-6):
    """Equivalent to RMSNorm(x + residual)."""
    rms = x.float().add(residual.float()).pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return ((x.float() + residual.float()) / rms * weight).to(x.dtype)


# --- KERNEL ---
@triton.jit
def _fused_rmsn_res_fwd(
    X_ptr, R_ptr, W_ptr, Y_ptr,
    stride_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    """One program per row. Fuses residual add + RMSNorm."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    # Load and add residual
    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    y = x + r

    # Compute sum of squares for RMS
    ss = y * y
    ss_sum = tl.sum(ss, axis=0)
    rms = tl.sqrt(ss_sum / D + eps)
    inv_rms = 1.0 / rms

    y_norm = y * inv_rms

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    out = y_norm * w

    tl.store(Y_ptr + row * stride_row + offs, out.to(x.dtype), mask=mask)


@triton.jit
def _fused_rmsn_res_bwd(
    X_ptr, R_ptr, W_ptr, DY_ptr, DX_ptr, DR_ptr, DW_ptr,
    stride_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N,
    eps: tl.constexpr,
):
    """Backward: computes dX, dResidual, dWeight."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    y = x + r

    # Compute RMS
    ss = y * y
    ss_sum = tl.sum(ss, axis=0)
    rms2 = ss_sum / D + eps
    rms = tl.sqrt(rms2)
    inv_rms = 1.0 / rms

    y_hat = y * inv_rms
    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    dy = tl.load(DY_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm backward:
    # d_norm = dy * w
    # d_rms = -sum(d_norm * y) / (rms^3)
    d_norm = dy * w
    d_rms = -tl.sum(d_norm * y, axis=0) / (rms2 * rms)
    # dx = d_norm / rms + d_rms * 2*y / D
    dx = d_norm * inv_rms + d_rms * 2.0 * y / D
    dx = tl.where(mask, dx, 0.0)

    tl.store(DX_ptr + row * stride_row + offs, dx.to(x.dtype), mask=mask)
    tl.store(DR_ptr + row * stride_row + offs, dx.to(x.dtype), mask=mask)
    tl.atomic_add(DW_ptr + offs, tl.where(mask, d_norm * inv_rms, 0.0))


class FusedRMSNormResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, eps=1e-6):
        orig_shape = x.shape
        D = x.shape[-1]
        N = x.numel() // D
        BLOCK_D = triton.next_power_of_2(D)

        x_c = x.contiguous()
        r_c = residual.contiguous()
        y = torch.empty_like(x_c)

        _fused_rmsn_res_fwd[(N,)](
            x_c, r_c, weight.contiguous(), y,
            D,
            D=D, BLOCK_D=BLOCK_D, eps=eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        ctx.save_for_backward(x_c, r_c, weight)
        ctx.eps = eps
        ctx.D = D
        ctx.BLOCK_D = BLOCK_D
        ctx.orig_shape = orig_shape
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_c, r_c, weight = ctx.saved_tensors
        eps = ctx.eps
        D = ctx.D
        BLOCK_D = ctx.BLOCK_D
        N = x_c.numel() // D

        dy_c = dy.contiguous().view(N, D)
        x_flat = x_c.view(N, D)
        r_flat = r_c.view(N, D)

        dx = torch.empty_like(x_flat)
        dr = torch.empty_like(x_flat)
        dw = torch.zeros(D, dtype=torch.float32, device=x_c.device)

        _fused_rmsn_res_bwd[(N,)](
            x_flat, r_flat, weight, dy_c, dx, dr, dw,
            D, D=D, BLOCK_D=BLOCK_D, N=N, eps=eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        return (
            dx.view(ctx.orig_shape),
            dr.view(ctx.orig_shape),
            dw.to(weight.dtype),
            None,
        )


def kernel_fn(x, residual, weight, eps=1e-6):
    return FusedRMSNormResidual.apply(x, residual, weight, eps)


def can_use_kernel(x, residual, weight, eps=1e-6):
    if not (x.is_cuda and residual.is_cuda):
        return False
    if x.shape != residual.shape:
        return False
    D = x.shape[-1]
    if D > 8192:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


SHAPES = {
    "vit_small": {"x": (2, 256, 384), "D": 384},
    "vit_l":     {"x": (2, 1024, 1024), "D": 1024},
    "vit_h":     {"x": (2, 2048, 1280), "D": 1280},
}
