"""Fused LayerNorm + Residual Add kernel.

Source: src/models/utils/modules.py:647-663 (Block.forward)
Pattern: x = x + drop_path(attn(norm1(x))); x = x + drop_path(mlp(norm2(x)))
Fuses: residual_add + layer_norm into one pass over the data.
Frequency: 2x per block x 24-32 blocks = 48-64 calls per forward pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, residual, weight, bias, eps=1e-5):
    """Equivalent to LayerNorm(x + residual)."""
    return torch.nn.functional.layer_norm(
        x + residual, (x.shape[-1],), weight, bias, eps
    )


# --- KERNEL ---
@triton.jit
def _fused_ln_res_fwd(
    X_ptr, R_ptr, W_ptr, B_ptr, Y_ptr,
    stride_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    """One program per row. Fuses residual add + LayerNorm."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    y = x + r

    # Welford mean
    mean = tl.sum(y, axis=0) / D
    # Variance
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)

    y_norm = (y - mean) * inv_std

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = y_norm * w + b

    tl.store(Y_ptr + row * stride_row + offs, out.to(x.dtype), mask=mask)


@triton.jit
def _fused_ln_res_bwd(
    X_ptr, R_ptr, W_ptr, DY_ptr, DX_ptr, DR_ptr, DW_ptr, DB_ptr,
    stride_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N,
    eps: tl.constexpr,
):
    """Backward: computes dX, dResidual, dWeight, dBias."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    y = x + r
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / D
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    y_hat = diff * inv_std

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    dy = tl.load(DY_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # LN backward: d_y_hat = dy * w
    d_yhat = dy * w
    # dvar = sum(d_yhat * (y - mean) * -0.5 * inv_std^3)
    dvar = tl.sum(d_yhat * diff * (-0.5) * inv_std * inv_std * inv_std, axis=0)
    # dmean = sum(-d_yhat * inv_std) + dvar * sum(-2*(y-mean)) / D
    dmean = tl.sum(-d_yhat * inv_std, axis=0) + dvar * tl.sum(-2.0 * diff, axis=0) / D
    # dx = d_yhat * inv_std + dvar * 2*(y-mean)/D + dmean/D
    dx = d_yhat * inv_std + dvar * 2.0 * diff / D + dmean / D
    dx = tl.where(mask, dx, 0.0)

    tl.store(DX_ptr + row * stride_row + offs, dx.to(x.dtype), mask=mask)
    tl.store(DR_ptr + row * stride_row + offs, dx.to(x.dtype), mask=mask)
    tl.atomic_add(DW_ptr + offs, tl.where(mask, dy * y_hat, 0.0))
    tl.atomic_add(DB_ptr + offs, tl.where(mask, dy, 0.0))


class FusedLNResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, bias, eps=1e-5):
        orig_shape = x.shape
        D = x.shape[-1]
        N = x.numel() // D
        BLOCK_D = triton.next_power_of_2(D)

        x_c = x.contiguous()
        r_c = residual.contiguous()
        y = torch.empty_like(x_c)

        _fused_ln_res_fwd[(N,)](
            x_c, r_c, weight.contiguous(), bias.contiguous(), y,
            D,  # stride_row = D for row-major contiguous
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
        db = torch.zeros(D, dtype=torch.float32, device=x_c.device)

        _fused_ln_res_bwd[(N,)](
            x_flat, r_flat, weight, dy_c, dx, dr, dw, db,
            D, D=D, BLOCK_D=BLOCK_D, N=N, eps=eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        return (
            dx.view(ctx.orig_shape),
            dr.view(ctx.orig_shape),
            dw.to(weight.dtype),
            db.to(weight.dtype),
            None,
        )


def kernel_fn(x, residual, weight, bias, eps=1e-5):
    return FusedLNResidual.apply(x, residual, weight, bias, eps)


def can_use_kernel(x, residual, weight, bias, eps=1e-5):
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


# Realistic shapes: ViT-L (embed=1024), ViT-H (embed=1280), small ViT (embed=384)
SHAPES = {
    "vit_small": {"x": (2, 256, 384), "D": 384},
    "vit_l":     {"x": (2, 1024, 1024), "D": 1024},
    "vit_h":     {"x": (2, 2048, 1280), "D": 1280},
}
