"""Fused add + LayerNorm for two-branch residual blocks.

Pattern: LayerNorm(x + b1 + b2)
Fuses: x + b1 + b2 + LayerNorm into one pass over the data.
Two residual branches are accumulated in fp32 before the normalization.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, b1, b2, weight, bias, eps=1e-5):
    """Equivalent to LayerNorm(x + b1 + b2)."""
    return torch.nn.functional.layer_norm(x + b1 + b2, (x.shape[-1],), weight, bias, eps)


# --- KERNEL ---
@triton.jit
def _fused_add_norm_res_fwd(
    X_ptr, B1_ptr, B2_ptr, W_ptr, B_ptr, Y_ptr,
    stride_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    """One program per row. Fuses x + b1 + b2 + LayerNorm."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    y = x + b1 + b2

    # LayerNorm statistics in fp32
    mean = tl.sum(y, axis=0) / D
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)

    y_norm = diff * inv_std

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = y_norm * w + b

    tl.store(Y_ptr + row * stride_row + offs, out.to(x.dtype), mask=mask)


@triton.jit
def _fused_add_norm_res_bwd(
    X_ptr, B1_ptr, B2_ptr, W_ptr, B_ptr,
    DY_ptr, DX_ptr, DB1_ptr, DB2_ptr, DW_ptr, DB_grad_ptr,
    stride_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N,
    eps: tl.constexpr,
):
    """Backward: computes dX, dB1, dB2, dWeight, dBias."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    y = x + b1 + b2

    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / D
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    y_hat = diff * inv_std

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    dy = tl.load(DY_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # LN backward
    d_yhat = dy * w
    dvar = tl.sum(d_yhat * diff * (-0.5) * inv_std * inv_std * inv_std, axis=0)
    dmean = tl.sum(-d_yhat * inv_std, axis=0) + dvar * tl.sum(-2.0 * diff, axis=0) / D
    dx = d_yhat * inv_std + dvar * 2.0 * diff / D + dmean / D
    dx = tl.where(mask, dx, 0.0)

    # Both residual branches receive the same gradient (add backward)
    tl.store(DX_ptr + row * stride_row + offs, dx.to(x.dtype), mask=mask)
    tl.store(DB1_ptr + row * stride_row + offs, dx.to(x.dtype), mask=mask)
    tl.store(DB2_ptr + row * stride_row + offs, dx.to(x.dtype), mask=mask)
    tl.atomic_add(DW_ptr + offs, tl.where(mask, dy * y_hat, 0.0))
    tl.atomic_add(DB_grad_ptr + offs, tl.where(mask, dy, 0.0))


class FusedAddNormResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b1, b2, weight, bias, eps=1e-5):
        orig_shape = x.shape
        D = x.shape[-1]
        N = x.numel() // D
        BLOCK_D = triton.next_power_of_2(D)

        x_c = x.contiguous()
        b1_c = b1.contiguous()
        b2_c = b2.contiguous()
        y = torch.empty_like(x_c)

        _fused_add_norm_res_fwd[(N,)](
            x_c, b1_c, b2_c, weight.contiguous(), bias.contiguous(), y,
            D,
            D=D, BLOCK_D=BLOCK_D, eps=eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        ctx.save_for_backward(x_c, b1_c, b2_c, weight)
        ctx.eps = eps
        ctx.D = D
        ctx.BLOCK_D = BLOCK_D
        ctx.orig_shape = orig_shape
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_c, b1_c, b2_c, weight = ctx.saved_tensors
        eps = ctx.eps
        D = ctx.D
        BLOCK_D = ctx.BLOCK_D
        N = x_c.numel() // D

        dy_c = dy.contiguous().view(N, D)
        x_flat = x_c.view(N, D)
        b1_flat = b1_c.view(N, D)
        b2_flat = b2_c.view(N, D)

        dx = torch.empty_like(x_flat)
        db1 = torch.empty_like(b1_flat)
        db2 = torch.empty_like(b2_flat)
        dw = torch.zeros(D, dtype=torch.float32, device=x_c.device)
        dbias = torch.zeros(D, dtype=torch.float32, device=x_c.device)

        _fused_add_norm_res_bwd[(N,)](
            x_flat, b1_flat, b2_flat, weight, bias, dy_c,
            dx, db1, db2, dw, dbias,
            D, D=D, BLOCK_D=BLOCK_D, N=N, eps=eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        return (
            dx.view(ctx.orig_shape),
            db1.view(ctx.orig_shape),
            db2.view(ctx.orig_shape),
            dw.to(weight.dtype),
            dbias.to(bias.dtype),
            None,
        )


def kernel_fn(x, b1, b2, weight, bias, eps=1e-5):
    return FusedAddNormResidual.apply(x, b1, b2, weight, bias, eps)


def _has_valid_shape(x, b1, b2, weight, bias):
    return (
        x.ndim >= 1
        and x.shape == b1.shape == b2.shape
        and weight.ndim == 1
        and bias.ndim == 1
        and weight.shape == bias.shape == (x.shape[-1],)
    )


def can_use_kernel(x, b1, b2, weight, bias, eps=1e-5):
    if not (x.is_cuda and b1.is_cuda and b2.is_cuda and weight.is_cuda and bias.is_cuda):
        return False
    if not (x.shape == b1.shape == b2.shape):
        return False
    D = x.shape[-1]
    if D > 8192:
        return False
    if not _has_valid_shape(x, b1, b2, weight, bias):
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "b1": (2, 1024, 1024), "b2": (2, 1024, 1024), "D": 1024},
    "vit_h": {"x": (2, 2048, 1280), "b1": (2, 2048, 1280), "b2": (2, 2048, 1280), "D": 1280},
    "small": {"x": (8, 256, 384), "b1": (8, 256, 384), "b2": (8, 256, 384), "D": 384},
}
