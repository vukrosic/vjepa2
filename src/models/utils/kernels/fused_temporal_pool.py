"""Fused temporal pooling kernel.

Pools video tokens across the time dimension: [B, T, N, C] -> [B, T//factor, N, C].
Uses average pooling with optional rounding mode (ceil/floor).
Used in video ViT for temporal downsampling between stages.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, pool_size):
    """x: [B, T, N, C] -> [B, T//pool_size, N, C]."""
    B, T, N, C = x.shape
    T_out = T // pool_size
    x = x[:, :T_out * pool_size]  # truncate if not divisible
    return x.view(B, T_out, pool_size, N, C).mean(dim=2)


# --- KERNEL ---
@triton.jit
def _temporal_pool_fwd_kernel(
    X, OUT,
    B, T, N, C,
    pool_size: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Grid: (B, T_out, N). Each program handles one (b, t_out, n) output slot."""
    b = tl.program_id(0)
    t_out = tl.program_id(1)
    n = tl.program_id(2)

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Initialize acc as block-level tensor before loop
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for p in range(pool_size):
        t_in = t_out * pool_size + p
        # Block-level pointer arithmetic
        x_base = (b * T * N + t_in * N + n) * C
        x_ptrs = x_base + offs_c
        vals = tl.load(X + x_ptrs, mask=mask_c, other=0.0).to(tl.float32)
        acc = acc + vals

    acc = acc / pool_size
    out_base = (b * (T // pool_size) * N + t_out * N + n) * C
    out_ptrs = out_base + offs_c
    tl.store(OUT + out_ptrs, acc, mask=mask_c)


def kernel_fn(x, pool_size):
    """x: [B, T, N, C] -> [B, T//pool_size, N, C]."""
    B, T, N, C = x.shape
    T_out = T // pool_size
    x = x[:, :T_out * pool_size].contiguous()
    out = torch.empty(B, T_out, N, C, dtype=x.dtype, device=x.device)
    BLOCK_C = triton.next_power_of_2(C)
    _temporal_pool_fwd_kernel[(B, T_out, N)](
        x, out, B, T, N, C, pool_size=pool_size,
        BLOCK_C=BLOCK_C,
        num_warps=min(8, max(1, BLOCK_C // 32)),
    )
    return out


def can_use_kernel(x, pool_size):
    if not x.is_cuda:
        return False
    if x.ndim != 4:
        return False
    if x.shape[1] % pool_size != 0:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


SHAPES = {
    "vit_l_16f_2x":   {"x": (2, 16, 196, 1024), "pool_size": 2},
    "vit_l_8f_2x":    {"x": (2, 8, 196, 1024),   "pool_size": 2},
    "vit_l_16f_4x":   {"x": (2, 16, 196, 1024), "pool_size": 4},
}
