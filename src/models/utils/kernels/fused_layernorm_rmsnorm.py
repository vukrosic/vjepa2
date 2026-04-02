"""Fused LayerNorm + RMSNorm hybrid kernel.

Pattern: Applies LayerNorm to one part of the tensor and RMSNorm to another,
computing both normalizations in a single pass over the data.
Used for hybrid heads in V-JEPA 2 where different components need different normalizations.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x, ln_weight, ln_bias, rms_weight, split_idx, eps=1e-5):
    """Apply LayerNorm to first split_idx channels and RMSNorm to the rest."""
    x_ln = x[..., :split_idx]
    x_rms = x[..., split_idx:]
    ln_out = torch.nn.functional.layer_norm(x_ln, (x_ln.shape[-1],), ln_weight, ln_bias, eps)
    rms = x_rms.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    rms_out = (x_rms / rms * rms_weight).to(x.dtype)
    return torch.cat([ln_out, rms_out], dim=-1)


# --- KERNEL ---
@triton.jit
def _fused_ln_rms_fwd(
    X_ptr, LN_W_ptr, LN_B_ptr, RMS_W_ptr, Y_ptr,
    stride_row,
    D: tl.constexpr, LN_D: tl.constexpr, RMS_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    """One program per row. Computes LayerNorm on first LN_D dims and RMSNorm on rest."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    ln_mask = offs < LN_D
    rms_mask = offs >= LN_D
    rms_offs = offs - LN_D

    # Load full row
    x = tl.load(X_ptr + row * stride_row + offs, mask=offs < D, other=0.0).to(tl.float32)

    # LayerNorm part: mean subtraction + variance
    ln_x = tl.load(X_ptr + row * stride_row + offs, mask=ln_mask, other=0.0).to(tl.float32)
    ln_mean = tl.sum(ln_x, axis=0) / LN_D
    ln_diff = ln_x - ln_mean
    ln_var = tl.sum(ln_diff * ln_diff, axis=0) / LN_D
    ln_inv_std = 1.0 / tl.sqrt(ln_var + EPS)
    ln_norm = ln_diff * ln_inv_std
    ln_w = tl.load(LN_W_ptr + offs, mask=ln_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(LN_B_ptr + offs, mask=ln_mask, other=0.0).to(tl.float32)
    ln_out = ln_norm * ln_w + ln_b

    # RMSNorm part
    rms_x = tl.load(X_ptr + row * stride_row + LN_D + rms_offs, mask=rms_mask, other=0.0).to(tl.float32)
    rms_ss = rms_x * rms_x
    rms_sum = tl.sum(rms_ss, axis=0)
    rms = tl.sqrt(rms_sum / RMS_D + EPS)
    rms_inv = 1.0 / rms
    rms_norm = rms_x * rms_inv
    rms_w = tl.load(RMS_W_ptr + rms_offs, mask=rms_mask, other=1.0).to(tl.float32)
    rms_out = rms_norm * rms_w

    # Combine
    out = tl.where(ln_mask, ln_out, rms_out)
    tl.store(Y_ptr + row * stride_row + offs, out.to(tl.float16), mask=offs < D)


@triton.jit
def _fused_ln_rms_bwd(
    X_ptr, LN_W_ptr, LN_B_ptr, RMS_W_ptr, DY_ptr,
    DX_ptr, DLN_W_ptr, DLN_B_ptr, DRMS_W_ptr,
    stride_row,
    D: tl.constexpr, LN_D: tl.constexpr, RMS_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    """Backward for LayerNorm + RMSNorm hybrid."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    ln_mask = offs < LN_D
    rms_mask = offs >= LN_D
    rms_offs = offs - LN_D

    x = tl.load(X_ptr + row * stride_row + offs, mask=offs < D, other=0.0).to(tl.float32)
    dy = tl.load(DY_ptr + row * stride_row + offs, mask=offs < D, other=0.0).to(tl.float32)

    # LN backward
    ln_x = tl.load(X_ptr + row * stride_row + offs, mask=ln_mask, other=0.0).to(tl.float32)
    ln_mean = tl.sum(ln_x, axis=0) / LN_D
    ln_diff = ln_x - ln_mean
    ln_var = tl.sum(ln_diff * ln_diff, axis=0) / LN_D
    ln_inv_std = 1.0 / tl.sqrt(ln_var + EPS)
    ln_y_hat = ln_diff * ln_inv_std
    ln_w = tl.load(LN_W_ptr + offs, mask=ln_mask, other=1.0).to(tl.float32)
    ln_dy = tl.load(DY_ptr + row * stride_row + offs, mask=ln_mask, other=0.0).to(tl.float32)
    d_lnhat = ln_dy * ln_w
    d_lnvar = tl.sum(d_lnhat * ln_diff * (-0.5) * ln_inv_std * ln_inv_std * ln_inv_std, axis=0)
    d_lnmean = tl.sum(-d_lnhat * ln_inv_std, axis=0) + d_lnvar * tl.sum(-2.0 * ln_diff, axis=0) / LN_D
    d_lnx = d_lnhat * ln_inv_std + d_lnvar * 2.0 * ln_diff / LN_D + d_lnmean / LN_D
    d_lnx = tl.where(ln_mask, d_lnx, 0.0)

    # RMS backward
    rms_x = tl.load(X_ptr + row * stride_row + LN_D + rms_offs, mask=rms_mask, other=0.0).to(tl.float32)
    rms_ss = rms_x * rms_x
    rms_sum = tl.sum(rms_ss, axis=0)
    rms2 = rms_sum / RMS_D + EPS
    rms = tl.sqrt(rms2)
    rms_inv = 1.0 / rms
    rms_w = tl.load(RMS_W_ptr + rms_offs, mask=rms_mask, other=1.0).to(tl.float32)
    rms_dy = tl.load(DY_ptr + row * stride_row + LN_D + rms_offs, mask=rms_mask, other=0.0).to(tl.float32)
    d_rmsnorm = rms_dy * rms_w
    d_rms = -tl.sum(d_rmsnorm * rms_x, axis=0) / (rms2 * rms)
    d_rmsx = d_rmsnorm * rms_inv + d_rms * 2.0 * rms_x / RMS_D
    d_rmsx = tl.where(rms_mask, d_rmsx, 0.0)

    dx = d_lnx + d_rmsx
    tl.store(DX_ptr + row * stride_row + offs, dx.to(x.dtype), mask=offs < D)
    tl.atomic_add(DLN_W_ptr + offs, tl.where(ln_mask, ln_dy * ln_y_hat, 0.0))
    tl.atomic_add(DLN_B_ptr + offs, tl.where(ln_mask, ln_dy, 0.0))
    tl.atomic_add(DRMS_W_ptr + rms_offs, tl.where(rms_mask, rms_dy * rms_x * rms_inv, 0.0))


class FusedLayerNormRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ln_weight, ln_bias, rms_weight, split_idx, eps=1e-5):
        orig_shape = x.shape
        D = x.shape[-1]
        LN_D = split_idx
        RMS_D = D - split_idx
        N = x.numel() // D
        BLOCK_D = triton.next_power_of_2(D)

        x_c = x.contiguous()
        y = torch.empty_like(x_c)

        _fused_ln_rms_fwd[(N,)](
            x_c, ln_weight.contiguous(), ln_bias.contiguous(), rms_weight.contiguous(), y,
            D,
            D=D, LN_D=LN_D, RMS_D=RMS_D, BLOCK_D=BLOCK_D, EPS=eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        ctx.save_for_backward(x_c, ln_weight, ln_bias, rms_weight)
        ctx.eps = eps
        ctx.D = D
        ctx.LN_D = LN_D
        ctx.RMS_D = RMS_D
        ctx.BLOCK_D = BLOCK_D
        ctx.orig_shape = orig_shape
        return y

    @staticmethod
    def backward(ctx, dy):
        x_c, ln_weight, ln_bias, rms_weight = ctx.saved_tensors
        D = ctx.D
        LN_D = ctx.LN_D
        RMS_D = ctx.RMS_D
        BLOCK_D = ctx.BLOCK_D
        N = x_c.numel() // D

        dy_c = dy.contiguous()
        dx = torch.empty_like(dy_c)
        d_ln_w = torch.zeros(LN_D, dtype=torch.float32, device=x_c.device)
        d_ln_b = torch.zeros(LN_D, dtype=torch.float32, device=x_c.device)
        d_rms_w = torch.zeros(RMS_D, dtype=torch.float32, device=x_c.device)

        _fused_ln_rms_bwd[(N,)](
            x_c, ln_weight, ln_bias, rms_weight, dy_c, dx,
            d_ln_w, d_ln_b, d_rms_w,
            D,
            D=D, LN_D=LN_D, RMS_D=RMS_D, BLOCK_D=BLOCK_D, EPS=ctx.eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        return (
            dx,
            d_ln_w.to(ln_weight.dtype),
            d_ln_b.to(ln_bias.dtype),
            d_rms_w.to(rms_weight.dtype),
            None,
            None,
        )


def kernel_fn(x, ln_weight, ln_bias, rms_weight, split_idx, eps=1e-5):
    return FusedLayerNormRMSNorm.apply(x, ln_weight, ln_bias, rms_weight, split_idx, eps)


def can_use_kernel(x, ln_weight, ln_bias, rms_weight, split_idx, eps=1e-5):
    if not x.is_cuda:
        return False
    D = x.shape[-1]
    if D > 8192:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if split_idx <= 0 or split_idx >= D:
        return False
    return True


SHAPES = {
    "vit_small": {"x": (2, 256, 384), "LN_D": 192, "RMS_D": 192},
    "vit_l":     {"x": (2, 1024, 1024), "LN_D": 512, "RMS_D": 512},
    "vit_h":     {"x": (2, 2048, 1280), "LN_D": 640, "RMS_D": 640},
}
