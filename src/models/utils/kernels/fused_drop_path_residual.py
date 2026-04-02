"""Fused DropPath + Residual Add kernel.

Source: src/models/utils/modules.py:662-663 (Block.forward)
Pattern: x = x + drop_path(y, drop_prob, self.training)
Fuses: stochastic depth mask generation + residual addition into one kernel.
DropPath drops entire samples (rows) from a batch — the mask is scalar per sample.
Frequency: 2x per block x 24-32 blocks = 48-64 calls per forward pass.
"""
import torch
import triton
import triton.language as tl


def _timm_drop_path_gpu(x, drop_prob, training):
    """Inline drop_path implementation (matches timm behavior)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.empty(shape, dtype=torch.bool, device=x.device).bernoulli_(keep_prob)
    x = x / keep_prob
    return x * mask


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, y, drop_prob, training):
    """x + drop_path(y) where drop_path uses per-sample stochastic depth."""
    return x + _timm_drop_path_gpu(y, drop_prob=drop_prob, training=training)


# --- KERNEL ---
@triton.jit
def _drop_path_residual_fwd(
    X_ptr, Y_ptr, OUT_ptr, KEEP_ptr,
    B, T,  # B=batch, T=tokens*features per sample (= N*C flattened)
    drop_p: tl.constexpr,
    scale: tl.constexpr,
    seed,
    BLOCK_T: tl.constexpr,
):
    """
    Grid: (B, T // BLOCK_T + 1)
    KEEP_ptr: [B] int8 mask (1=keep, 0=drop) for storing backward
    """
    b_idx = tl.program_id(0)
    t_pid = tl.program_id(1)
    offs_t = t_pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    # Generate per-sample keep mask using sample index as RNG offset
    # tl.rand gives float in [0,1); compare against drop_p
    keep_float = tl.rand(seed, tl.arange(0, 1) + b_idx)
    keep = keep_float > drop_p

    # Store keep mask (first thread block per row writes it)
    if t_pid == 0:
        tl.store(KEEP_ptr + b_idx, keep.to(tl.int8))

    x = tl.load(X_ptr + b_idx * T + offs_t, mask=mask_t, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + b_idx * T + offs_t, mask=mask_t, other=0.0).to(tl.float32)

    # y_dropped = y * keep * scale
    y_scaled = tl.where(keep, y * scale, 0.0)
    out = x + y_scaled

    tl.store(OUT_ptr + b_idx * T + offs_t, out.to(x.dtype), mask=mask_t)


@triton.jit
def _drop_path_residual_bwd(
    DY_ptr, DX_ptr, DRES_ptr, KEEP_ptr,
    B, T,
    scale: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    b_idx = tl.program_id(0)
    t_pid = tl.program_id(1)
    offs_t = t_pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    dy = tl.load(DY_ptr + b_idx * T + offs_t, mask=mask_t, other=0.0).to(tl.float32)
    keep = tl.load(KEEP_ptr + b_idx).to(tl.int1)

    # dx = dy (residual passes gradient through unchanged)
    tl.store(DX_ptr + b_idx * T + offs_t, dy.to(dy.dtype), mask=mask_t)

    # d_residual = dy * keep * scale
    d_res = tl.where(keep, dy * scale, 0.0)
    tl.store(DRES_ptr + b_idx * T + offs_t, d_res.to(dy.dtype), mask=mask_t)


class FusedDropPathResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, drop_prob, training):
        if not training or drop_prob == 0.0:
            out = x + y
            ctx.save_for_backward(torch.ones(x.shape[0], dtype=torch.int8, device=x.device))
            ctx.drop_prob = 0.0
            ctx.scale = 1.0
            ctx.shape = x.shape
            return out

        B = x.shape[0]
        T = x.numel() // B
        scale = 1.0 / (1.0 - drop_prob)
        BLOCK_T = 1024

        x_c = x.contiguous().view(B, T)
        y_c = y.contiguous().view(B, T)
        out = torch.empty_like(x_c)
        keep_mask = torch.empty(B, dtype=torch.int8, device=x.device)

        seed = torch.randint(0, 2**31, (1,), device="cpu").item()
        n_t_blocks = (T + BLOCK_T - 1) // BLOCK_T
        _drop_path_residual_fwd[(B, n_t_blocks)](
            x_c, y_c, out, keep_mask,
            B, T, drop_p=drop_prob, scale=scale, seed=seed,
            BLOCK_T=BLOCK_T, num_warps=4,
        )
        ctx.save_for_backward(keep_mask)
        ctx.drop_prob = drop_prob
        ctx.scale = scale
        ctx.shape = x.shape
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad_out):
        (keep_mask,) = ctx.saved_tensors
        shape = ctx.shape
        B = shape[0]
        T = grad_out.numel() // B
        BLOCK_T = 1024

        dy = grad_out.contiguous().view(B, T)
        dx = torch.empty_like(dy)
        d_res = torch.empty_like(dy)

        n_t_blocks = (T + BLOCK_T - 1) // BLOCK_T
        _drop_path_residual_bwd[(B, n_t_blocks)](
            dy, dx, d_res, keep_mask,
            B, T, scale=ctx.scale, BLOCK_T=BLOCK_T, num_warps=4,
        )
        return dx.view(shape), d_res.view(shape), None, None


def kernel_fn(x, y, drop_prob=0.1, training=True):
    return FusedDropPathResidual.apply(x, y, drop_prob, training)


def can_use_kernel(x, y, drop_prob=0.1, training=True):
    if not (x.is_cuda and y.is_cuda):
        return False
    if x.shape != y.shape:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


# ViT-L: [B, N, C] where C=1024; 2x per block, 24 blocks
SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "y": (2, 1024, 1024)},
    "vit_h": {"x": (2, 2048, 1280), "y": (2, 2048, 1280)},
    "small": {"x": (8, 256, 384), "y": (8, 256, 384)},
}
