"""Fused V-JEPA 2 Lp loss kernel.

Source: app/vjepa_2_1/train.py:641
Pattern: torch.mean(torch.abs(zij - h_term) ** loss_exp) / loss_exp
Fuses: subtract, abs, pow, mean into single-pass reduction.
Avoids materializing 3 intermediate tensors.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(z, h, loss_exp):
    return torch.mean(torch.abs(z - h) ** loss_exp) / loss_exp


# --- KERNEL ---
@triton.jit
def _lp_loss_kernel(
    Z, H, OUT,
    N: tl.constexpr,
    loss_exp: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Block-level partial sum. Final reduction done in Python."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    z = tl.load(Z + offs, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(H + offs, mask=mask, other=0.0).to(tl.float32)

    diff = z - h
    abs_diff = tl.where(diff >= 0, diff, -diff)

    if loss_exp == 1.0:
        val = abs_diff
    elif loss_exp == 2.0:
        val = abs_diff * abs_diff
    else:
        val = tl.math.pow(abs_diff, loss_exp)

    # Masked sum
    val = tl.where(mask, val, 0.0)
    block_sum = tl.sum(val, axis=0)
    tl.store(OUT + pid, block_sum)


def kernel_fn(z, h, loss_exp):
    assert z.is_contiguous() and h.is_contiguous()
    N = z.numel()
    BLOCK = 1024
    n_blocks = (N + BLOCK - 1) // BLOCK
    partial_sums = torch.empty(n_blocks, dtype=torch.float32, device=z.device)
    _lp_loss_kernel[(n_blocks,)](z, h, partial_sums, N, loss_exp, BLOCK=BLOCK, num_warps=4)
    return partial_sums.sum() / (N * loss_exp)


def can_use_kernel(z, h, loss_exp):
    return (z.is_cuda and h.is_cuda and
            z.is_contiguous() and h.is_contiguous() and
            z.shape == h.shape and
            isinstance(loss_exp, (int, float)) and loss_exp > 0)


SHAPES = {
    "small_mask": {"z": (2, 256, 384), "h": (2, 256, 384), "loss_exp": 1.0},
    "medium_mask": {"z": (4, 512, 768), "h": (4, 512, 768), "loss_exp": 1.0},
    "large_l2": {"z": (2, 1024, 1024), "h": (2, 1024, 1024), "loss_exp": 2.0},
    "large_l1": {"z": (2, 1024, 1024), "h": (2, 1024, 1024), "loss_exp": 1.0},
}
