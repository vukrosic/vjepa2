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
def _fused_idx_select_mean_fwd(X, INDICES, Y, B: tl.constexpr, N: tl.constexpr, D: tl.constexpr, M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # One program per (B, D) — accumulates sum over M indices
    sum_val = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for m in range(M):
        idx = tl.load(INDICES + pid_b * M + m).to(tl.int32)
        x_row_ptr = pid_b * N * D + idx * D
        for d in range(D):
            val = tl.load(X + x_row_ptr + d, mask=mask_d, other=0.0).to(tl.float32)
            sum_val = sum_val + val
            mask_d = (offs_d + d + 1) < D
        mask_d = offs_d < D

    mean_val = sum_val / M
    for d in range(D):
        tl.store(Y + pid_b * D + d, mean_val, mask=mask_d)
        mask_d = (offs_d + d + 1) < D


def kernel_fn(x, indices):
    B, N, D = x.shape
    M = indices.shape[1]
    assert indices.is_contiguous()
    y = torch.empty(B, D, dtype=x.dtype, device=x.device)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B,)
    _fused_idx_select_mean_fwd[grid](x, indices, y, B, N, D, M, BLOCK_D=BLOCK_D, num_warps=4)
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
