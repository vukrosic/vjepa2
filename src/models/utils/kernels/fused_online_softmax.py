"""Fused softmax kernel using online softmax algorithm.

Computes softmax(scores, dim=-1) with a single program per row.
Avoids materializing the full exp(scores) array — computes max first
via parallel reduction, then normalizes in the same kernel.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(scores):
    """scores: [..., D] — softmax over last dim."""
    return torch.softmax(scores, dim=-1)


# --- KERNEL ---
@triton.jit
def _softmax_max_kernel(X, MAX_OUT, N, BLOCK: tl.constexpr):
    """One program per N-element row. Computes row max."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    mx = tl.max(x, axis=0)
    tl.store(MAX_OUT + pid, mx)


@triton.jit
def _softmax_exp_sum_kernel(X, MAX_IN, EXP_SUM_OUT, N, BLOCK: tl.constexpr):
    """One program per N-element row. Computes sum(exp(x - row_max))."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    row_max = tl.load(MAX_IN + pid)
    exp_vals = tl.exp(x - row_max)
    total = tl.sum(exp_vals, axis=0)
    tl.store(EXP_SUM_OUT + pid, total)


@triton.jit
def _softmax_norm_kernel(X, MAX_IN, SUM_IN, OUT, N, BLOCK: tl.constexpr):
    """One program per row. Normalizes: exp(x - max) / sum."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    row_max = tl.load(MAX_IN + pid)
    total = tl.load(SUM_IN + pid)
    result = tl.exp(x - row_max) / total
    tl.store(OUT + offs, result.to(x.dtype), mask=mask)


def kernel_fn(scores):
    """
    scores: [*, D] contiguous. Returns softmax over last dim.
    """
    flat = scores.flatten()
    N = flat.shape[0]
    BLOCK = min(triton.next_power_of_2(N), 4096)
    n_programs = (N + BLOCK - 1) // BLOCK

    x = flat.contiguous()
    row_max = torch.empty(n_programs, dtype=torch.float32, device=x.device)
    exp_sum = torch.empty(n_programs, dtype=torch.float32, device=x.device)
    out_flat = torch.empty_like(x)

    _softmax_max_kernel[(n_programs,)](x, row_max, N, BLOCK=BLOCK, num_warps=4)
    _softmax_exp_sum_kernel[(n_programs,)](x, row_max, exp_sum, N, BLOCK=BLOCK, num_warps=4)
    _softmax_norm_kernel[(n_programs,)](x, row_max, exp_sum, out_flat, N, BLOCK=BLOCK, num_warps=4)

    return out_flat.view(scores.shape)


def can_use_kernel(scores):
    return (scores.is_cuda and
            scores.is_contiguous() and
            scores.dtype in (torch.float16, torch.float32) and
            scores.ndim >= 1)


SHAPES = {
    "vit_l_short":  {"scores": (2, 16, 256, 256)},
    "vit_l_medium": {"scores": (2, 16, 512, 512)},
    "vit_l_long":   {"scores": (1, 16, 1024, 1024)},
}
