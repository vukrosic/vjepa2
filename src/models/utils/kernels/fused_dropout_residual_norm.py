"""Fused Dropout + Residual Add + LayerNorm kernel.

Pattern: LayerNorm(dropout(x) + residual)
Fuses: dropout mask + residual add + layer norm into one pass.
Eliminates two intermediate HBM round-trips (after dropout and after residual add).
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x, residual, weight, bias, p=0.1, training=True):
    """Equivalent to LayerNorm(F.dropout(x, p) + residual)."""
    if training:
        x = torch.nn.functional.dropout(x, p=p)
    x = x + residual
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias)


# --- KERNEL ---
@triton.jit
def _fused_dropout_residual_norm_fwd(
    X_ptr, R_ptr, W_ptr, B_ptr, Y_ptr, MASK_PTR,
    stride_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    P: tl.constexpr,
    EPS: tl.constexpr,
    SEED: tl.constexpr,
):
    """One program per row. Fuses dropout + residual add + LayerNorm."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    # Load x and residual
    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # Dropout: generate random mask using simple hash-based approach
    # For simplicity we use block-level random - each element has probability P of being dropped
    rng_offset = row * D + offs
    # Simple hash-based pseudo-random for dropout mask
    rand_val = tl.abs(tl.math.hash(tl.cast(rng_offset + SEED, tl.int32), 0)) / 4294967296.0
    keep_mask = rand_val > P
    x_drop = tl.where(keep_mask, x / (1.0 - P), 0.0)

    y = x_drop + r

    # LayerNorm
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / D
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + EPS)

    y_norm = diff * inv_std

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = y_norm * w + b

    tl.store(Y_ptr + row * stride_row + offs, out.to(x.dtype), mask=mask)
    # Store dropout mask for backward
    tl.store(MASK_PTR + row * stride_row + offs, tl.cast(keep_mask, tl.int8), mask=mask)


@triton.jit
def _fused_dropout_residual_norm_bwd(
    X_ptr, R_ptr, W_ptr, DY_ptr, MASK_PTR, DX_ptr, DR_ptr, DW_ptr, DB_ptr,
    stride_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N,
    P: tl.constexpr,
    EPS: tl.constexpr,
):
    """Backward: computes dX, dResidual, dWeight, dBias."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    keep_mask = tl.load(MASK_PTR + row * stride_row + offs).to(tl.int32)

    rng_offset = row * D + offs
    rand_val = tl.abs(tl.math.hash(tl.cast(rng_offset, tl.int32), 0)) / 4294967296.0
    keep = rand_val > P

    x_drop = tl.where(keep, x / (1.0 - P), 0.0)
    y = x_drop + r

    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / D
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + EPS)
    y_hat = diff * inv_std

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    dy = tl.load(DY_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    d_yhat = dy * w
    dvar = tl.sum(d_yhat * diff * (-0.5) * inv_std * inv_std * inv_std, axis=0)
    dmean = tl.sum(-d_yhat * inv_std, axis=0) + dvar * tl.sum(-2.0 * diff, axis=0) / D
    dy_drop = d_yhat * inv_std + dvar * 2.0 * diff / D + dmean / D
    dy_drop = tl.where(mask, dy_drop, 0.0)

    # dx = dy_drop * (keep / (1-P))
    dx = tl.where(keep, dy_drop / (1.0 - P), 0.0)
    tl.store(DX_ptr + row * stride_row + offs, dx.to(x.dtype), mask=mask)
    tl.store(DR_ptr + row * stride_row + offs, dy_drop.to(x.dtype), mask=mask)
    tl.atomic_add(DW_ptr + offs, tl.where(mask, dy * y_hat, 0.0))
    tl.atomic_add(DB_ptr + offs, tl.where(mask, dy, 0.0))


class FusedDropoutResidualNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, bias, p=0.1, training=True, seed=0):
        orig_shape = x.shape
        D = x.shape[-1]
        N = x.numel() // D
        BLOCK_D = triton.next_power_of_2(D)

        x_c = x.contiguous()
        r_c = residual.contiguous()
        y = torch.empty_like(x_c)
        mask = torch.empty(N, D, dtype=torch.int8, device=x_c.device)

        _fused_dropout_residual_norm_fwd[(N,)](
            x_c, r_c, weight.contiguous(), bias.contiguous(), y, mask,
            D,
            D=D, BLOCK_D=BLOCK_D, P=p, EPS=1e-5, SEED=seed,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        ctx.save_for_backward(x_c, r_c, weight)
        ctx.p = p
        ctx.eps = 1e-5
        ctx.D = D
        ctx.BLOCK_D = BLOCK_D
        ctx.orig_shape = orig_shape
        ctx.seed = seed
        ctx.mask = mask
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_c, r_c, weight = ctx.saved_tensors
        p = ctx.p
        D = ctx.D
        BLOCK_D = ctx.BLOCK_D
        N = x_c.numel() // D
        mask = ctx.mask

        dy_c = dy.contiguous().view(N, D)
        x_flat = x_c.view(N, D)
        r_flat = r_c.view(N, D)

        dx = torch.empty_like(x_flat)
        dr = torch.empty_like(x_flat)
        dw = torch.zeros(D, dtype=torch.float32, device=x_c.device)
        db = torch.zeros(D, dtype=torch.float32, device=x_c.device)

        _fused_dropout_residual_norm_bwd[(N,)](
            x_flat, r_flat, weight, dy_c, mask, dx, dr, dw, db,
            D, D=D, BLOCK_D=BLOCK_D, N=N, P=p, EPS=ctx.eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        return (
            dx.view(ctx.orig_shape),
            dr.view(ctx.orig_shape),
            dw.to(weight.dtype),
            db.to(bias.dtype),
            None,
            None,
            None,
        )


def kernel_fn(x, residual, weight, bias, p=0.1, training=True, seed=0):
    return FusedDropoutResidualNorm.apply(x, residual, weight, bias, p, training, seed)


def can_use_kernel(x, residual, weight, bias, p=0.1, training=True):
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
