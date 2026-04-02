"""Fused Sum Square kernel.

Pattern: returns (sum(x), sum(x^2)) tuple
Fuses: square + sum into one reduction pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return x.sum(), x.pow(2).sum()


# --- KERNEL ---
@triton.jit
def _sum_square_kernel(X, OUT_SUM, OUT_SQ, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    val_sum = tl.where(mask, x, 0.0)
    val_sq = tl.where(mask, x * x, 0.0)
    block_sum = tl.sum(val_sum, axis=0)
    block_sq = tl.sum(val_sq, axis=0)
    tl.store(OUT_SUM + pid, block_sum)
    tl.store(OUT_SQ + pid, block_sq)


def kernel_fn(x):
    assert x.is_contiguous()
    N = x.numel()
    BLOCK = 1024
    n_blocks = (N + BLOCK - 1) // BLOCK
    partial_sum = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    partial_sq = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    _sum_square_kernel[(n_blocks,)](x, partial_sum, partial_sq, N, BLOCK=BLOCK, num_warps=4)
    return partial_sum.sum(), partial_sq.sum()


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
