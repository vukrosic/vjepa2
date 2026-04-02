"""Fused mask fill + softmax kernel for attention.

Applies -inf mask inline then computes softmax in one pass — avoids
materializing the masked intermediate. One Triton program per (b, h, row).
Used in V-JEPA 2 attention layers to fuse mask application with softmax.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(scores, mask, scale=1.0):
    scores = scores * scale
    scores = scores.masked_fill(~mask, float('-inf'))
    return torch.softmax(scores, dim=-1)


# --- KERNEL ---
@triton.jit
def _fused_masked_fill_softmax_fwd(
    SCORES, MASK, OUT,
    scale,
    stride_b, stride_h, stride_n,
    B, H, N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per (b, h, row).

    Decomposes pid -> (b, h, n):
      pid = b*H*N + h*N + n  (each pid = one row of the attention matrix)
    mask is (B, 1, N, N), broadcasted across heads.
    mask[b, 0, n, j] is at linear offset: b*N*N + n*N + j
    """
    pid = tl.program_id(0)
    # Decompose: pid = b*H*N + h*N + n
    n = pid % N
    tmp = pid // N
    h = tmp % H
    b = pid // (H * N)

    # Mask base: mask[b, 0, n, 0] at linear offset b*N*N + n*N
    mask_base = (b * N + n) * N

    # Scores row base: scores[b, h, n, 0] at linear offset
    row_base = b * stride_b + h * stride_h + n * stride_n

    # --- Online softmax: find max, applying mask inline ---
    m = -float("inf")
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask_col = cols < N
        s = tl.load(SCORES + row_base + cols, mask=mask_col, other=0.0).to(tl.float32)
        s = s * scale
        mask_val = tl.load(MASK + mask_base + cols, mask=mask_col, other=0).to(tl.float32)
        s = tl.where(mask_val > 0, s, -float("inf"))
        m = tl.maximum(m, tl.max(s, axis=0))

    # --- Accumulate exp sum ---
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask_col = cols < N
        s = tl.load(SCORES + row_base + cols, mask=mask_col, other=0.0).to(tl.float32)
        s = s * scale
        mask_val = tl.load(MASK + mask_base + cols, mask=mask_col, other=0).to(tl.float32)
        # Apply mask BEFORE subtract-m so masked positions get exp(-inf)=0
        s = tl.where(mask_val > 0, s, -float("inf"))
        s = tl.exp(s - m)
        acc += tl.where(mask_col, s, 0.0)
    denom = tl.sum(acc, axis=0)

    # --- Normalize and store ---
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask_col = cols < N
        s = tl.load(SCORES + row_base + cols, mask=mask_col, other=0.0).to(tl.float32)
        s = s * scale
        mask_val = tl.load(MASK + mask_base + cols, mask=mask_col, other=0).to(tl.float32)
        s = tl.where(mask_val > 0, s, -float("inf"))
        s = tl.exp(s - m) / denom
        tl.store(OUT + row_base + cols, s, mask=mask_col)


def kernel_fn(scores, mask, scale=1.0):
    """scores: [B, H, N, N], mask: [B, 1, N, N] (bool, True=keep, False=mask).
    Returns: [B, H, N, N] softmax probabilities.
    """
    assert scores.is_contiguous()
    B, H, N, N2 = scores.shape
    assert N == N2, "scores must be square in last two dims"
    assert mask.shape == (B, 1, N, N), "mask shape mismatch"
    BLOCK_N = min(triton.next_power_of_2(N), 4096)
    # Convert bool mask to int8 — tl.load(..., other=0.0) on a bool pointer is illegal
    mask_int = mask.to(torch.int8)
    out = torch.empty_like(scores)
    total_rows = B * H * N
    _fused_masked_fill_softmax_fwd[(total_rows,)](
        scores, mask_int, out,
        float(scale),
        scores.stride(0), scores.stride(1), scores.stride(2),
        B, H, N=N, BLOCK_N=BLOCK_N,
        num_warps=min(16, max(1, BLOCK_N // 32)),
    )
    return out


def can_use_kernel(scores, mask, scale):
    return (
        scores.is_cuda
        and mask.is_cuda
        and scores.is_contiguous()
        and mask.is_contiguous()
        and scores.ndim == 4
        and scores.shape[-1] == scores.shape[-2]
        and scores.shape[0] == mask.shape[0]
        and scores.shape[2] == mask.shape[2]
        and mask.shape[1] == 1
        and scores.dtype in (torch.float16, torch.float32)
    )


SHAPES = {
    "seq256":  {"scores": (2, 16, 256, 256), "scale": 0.125},
    "seq512":  {"scores": (2, 16, 512, 512), "scale": 0.125},
    "seq1024": {"scores": (1, 16, 1024, 1024), "scale": 0.125},
}
