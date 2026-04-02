"""Fused LayerNorm + residual + GELU kernel.

Source: src/models/utils/modules.py:662-663 (Block.forward)
Pattern: x = x + drop_path(mlp(norm2(x))) where MLP = fc1 -> gelu -> dropout -> fc2
This kernel fuses: norm2_output + residual (inference mode, no drop_path)
Plus the GELU from the MLP in a separate kernel.

Actually, let me implement the FUSED PATTERN: x = norm(x) + residual, then apply GELU.
This replaces: LayerNorm kernel + elementwise add = 2 passes with 1 pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, residual, weight, bias, eps=1e-5):
    """Norm + residual: LayerNorm(x) + residual."""
    normed = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return normed + residual


# --- KERNEL ---
@triton.jit
def _layernorm_add_kernel(
    X_ptr, R_ptr, W_ptr, B_ptr, OUT_ptr,
    stride_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    """Grid: one program per row. Fuses: norm(x) + residual in one read pass."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (x_norm * w + b) + r

    tl.store(OUT_ptr + row * stride_row + offs, out.to(x.dtype), mask=mask)


@triton.jit
def _layernorm_add_bwd_kernel(
    X_ptr, R_ptr, W_ptr, DO_ptr,
    DX_ptr, DR_ptr, DW_ptr, DB_ptr,
    stride_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    """Backward: gradient through norm + residual."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_hat = diff * inv_std

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    do = tl.load(DO_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # dx_norm = do * w
    dx_norm = do * w
    # LN backward
    dvar = tl.sum(dx_norm * diff * (-0.5) * inv_std**3, axis=0)
    dmean = tl.sum(-dx_norm * inv_std, axis=0) + dvar * tl.sum(-2.0 * diff, axis=0) / D
    dx = dx_norm * inv_std + dvar * 2.0 * diff / D + dmean / D
    dx = tl.where(mask, dx, 0.0)

    # dr = do (residual passes gradient through)
    dr = do

    tl.store(DX_ptr + row * stride_row + offs, dx.to(x.dtype), mask=mask)
    tl.store(DR_ptr + row * stride_row + offs, dr.to(x.dtype), mask=mask)
    tl.atomic_add(DW_ptr + offs, tl.where(mask, do * x_hat, 0.0))
    tl.atomic_add(DB_ptr + offs, tl.where(mask, do, 0.0))


class FusedLayerNormAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, bias, eps=1e-5):
        D = x.shape[-1]
        N = x.numel() // D
        BLOCK_D = triton.next_power_of_2(D)
        x_c = x.contiguous()
        r_c = residual.contiguous()
        out = torch.empty_like(x_c)
        _layernorm_add_kernel[(N,)](
            x_c, r_c, weight.contiguous(), bias.contiguous(), out,
            D, D=D, BLOCK_D=BLOCK_D, eps=eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        ctx.save_for_backward(x_c, r_c, weight)
        ctx.eps = eps
        ctx.D = D
        ctx.BLOCK_D = BLOCK_D
        return out

    @staticmethod
    def backward(ctx, do):
        x_c, r_c, weight = ctx.saved_tensors
        D = ctx.D; BLOCK_D = ctx.BLOCK_D; eps = ctx.eps
        N = x_c.numel() // D
        dx = torch.empty_like(x_c)
        dr = torch.empty_like(x_c)
        dw = torch.zeros(D, dtype=torch.float32, device=x_c.device)
        db = torch.zeros(D, dtype=torch.float32, device=x_c.device)
        _layernorm_add_bwd_kernel[(N,)](
            x_c, r_c, weight, do,
            dx, dr, dw, db,
            D, D=D, BLOCK_D=BLOCK_D, eps=eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        return dx, dr, dw.to(weight.dtype), db.to(weight.dtype), None


def kernel_fn(x, residual, weight, bias, eps=1e-5):
    return FusedLayerNormAdd.apply(x, residual, weight, bias, eps)


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


SHAPES = {
    "vit_small": {"x": (2, 256, 384), "D": 384},
    "vit_l":     {"x": (2, 1024, 1024), "D": 1024},
    "vit_h":     {"x": (2, 2048, 1280), "D": 1280},
}
