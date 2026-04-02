"""Fused scale + softmax kernel for attention.

Computes softmax(scores * scale, dim=-1) without materializing the scaled
intermediate. One Triton program per (b, h, row) in the [B*H*N] grid.
Used in V-JEPA 2 attention layers to reduce memory bandwidth.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(scores, scale):
    """scores: [B, H, N, N] raw attention scores."""
    return torch.softmax(scores * scale, dim=-1)


# --- KERNEL ---
@triton.jit
def _fused_softmax_temperature_fwd(
    SCORES, OUT,
    scale,
    stride_b, stride_h, stride_n,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # pid maps to (b, h, row)
    pid = tl.program_id(0)
    row_ptr = SCORES + pid * stride_n
    out_ptr = OUT + pid * stride_n

    # --- Online softmax: find max ---
    m = -float("inf")
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        s = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        s = s * scale
        m = tl.maximum(m, tl.max(s, axis=0))

    # --- Accumulate exp sum ---
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        s = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        s = s * scale - m
        acc += tl.where(mask, tl.exp(s), 0.0)
    denom = tl.sum(acc, axis=0)

    # --- Normalize and store ---
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        s = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        s = tl.exp(s * scale - m) / denom
        tl.store(out_ptr + cols, s, mask=mask)


def kernel_fn(scores, scale):
    assert scores.is_contiguous()
    B, H, N, N2 = scores.shape
    assert N == N2, "scores must be square in last two dims"
    BLOCK_N = min(triton.next_power_of_2(N), 4096)
    out = torch.empty_like(scores)
    total_rows = B * H * N
    _fused_softmax_temperature_fwd[(total_rows,)](
        scores, out,
        float(scale),
        scores.stride(0), scores.stride(1), scores.stride(2),
        N=N, BLOCK_N=BLOCK_N,
        num_warps=min(16, max(1, BLOCK_N // 32)),
    )
    return out


def can_use_kernel(scores, scale):
    return (
        scores.is_cuda
        and scores.is_contiguous()
        and scores.ndim == 4
        and scores.shape[-1] == scores.shape[-2]
        and scores.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


SHAPES = {
    "vit_l_s": {"scores": (2, 16, 256, 256), "scale": 0.125},
    "vit_l_m": {"scores": (2, 16, 512, 512), "scale": 0.125},
    "vit_l_l": {"scores": (1, 16, 1024, 1024), "scale": 0.125},
}
