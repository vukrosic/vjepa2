"""Warp-level parallel reduction kernel.

Demonstrates Triton warp-shuffle reductions for fast intra-warp sums.
Useful for kernels that need fast partial reductions (e.g., gradient norm
computation, loss reduction).

Baseline: `tensor.sum()` — single-threaded on CPU or slow GPU kernel.
Kernel: uses warp shuffle (`tl.shuffle_xor`) for O(log32) instead of O(N) reduction.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    """x: [N] — sum all elements."""
    return x.sum()


# --- KERNEL ---
@triton.jit
def _warp_reduce_sum_kernel(X, OUT, N, BLOCK: tl.constexpr):
    """Grid: (N // BLOCK + 1,) — each program reduces BLOCK elements."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    val = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)

    # Warp-level reduction using shuffle
    width = 16
    for _ in range(5):  # log2(32) = 5 steps
        val = val + tl.shuffle_xor(val, width)
        width = width // 2

    # First lane of each warp writes the result
    if tl.lane_id() == 0:
        tl.store(OUT + pid, val)


def kernel_fn(x):
    """Sum all elements of x using warp-level reduction."""
    N = x.numel()
    BLOCK = 256
    n_programs = (N + BLOCK - 1) // BLOCK

    x_c = x.contiguous().flatten()
    partial = torch.empty(n_programs, dtype=torch.float32, device=x_c.device)

    _warp_reduce_sum_kernel[(n_programs,)](x_c, partial, N, BLOCK=BLOCK, num_warps=8)

    # Final reduction on CPU (small array)
    return partial.sum()


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l_grad":   {"x": (1024 * 1024,),},  # ~1M element gradient
    "vit_h_grad":   {"x": (5120 * 1280,),},  # ~6.5M
    "medium_grad":  {"x": (4096 * 1024,),},
}
