"""Fused index select + mean pooling kernel.

Pattern: out = mean of rows selected by indices from x.
Fuses: gather + mean in one pass, avoiding materialization of gathered tensor.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x, indices):
    """x: [B, N, D], indices: [B, M] — select M tokens, return mean"""
    B, N, D = x.shape
    gathered = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, D))
    return gathered.mean(dim=1)


# --- KERNEL ---
@triton.jit
def _fused_idx_select_mean_fwd(
    X, INDICES, Y,
    B: tl.constexpr, N: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Grid: (B, D) — one program per (b, d) position.

    Loads M indices for batch b using block-level indexing,
    then loops over blocks loading x values and accumulating sum.
    Finally stores mean = sum / M at position (b, d).
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Load M indices for this batch
    idx_block = tl.load(INDICES + pid_b * M + offs_m, mask=mask_m, other=0).to(tl.int32)

    # Accumulate sum over M indices for this (b, d)
    acc = 0.0
    for m in range(M):
        idx = tl.load(INDICES + pid_b * M + m)
        x_ptr = pid_b * N * D + idx * D + pid_d
        val = tl.load(X + x_ptr).to(tl.float32)
        acc += val

    mean_val = acc / M
    tl.store(Y + pid_b * D + pid_d, mean_val)


def kernel_fn(x, indices):
    B, N, D = x.shape
    M = indices.shape[1]
    assert indices.is_contiguous()
    y = torch.empty(B, D, dtype=x.dtype, device=x.device)
    BLOCK_M = triton.next_power_of_2(M)
    grid = (B, D)
    _fused_idx_select_mean_fwd[grid](
        x, indices, y, B, N, D, M, BLOCK_M=BLOCK_M, num_warps=4,
    )
    return y


def can_use_kernel(x, indices):
    return (x.is_cuda and indices.is_cuda and
            x.is_contiguous() and indices.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            indices.dtype == torch.long)


SHAPES = {
    "small":   {"x": (2, 784, 384),   "indices_shape": (2, 196)},
    "medium":  {"x": (2, 1568, 1024), "indices_shape": (2, 512)},
    "large":   {"x": (2, 3136, 1280), "indices_shape": (2, 1024)},
}
