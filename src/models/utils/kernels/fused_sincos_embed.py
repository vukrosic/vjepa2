"""Fused sincos position embedding kernel.

Source: src/models/utils/pos_embs.py
Pattern: CPU np.sin/np.cos grid then .to(device)
Fuses: Generates sincos embeddings directly on GPU, avoiding CPU-GPU transfer.
The kernel computes sin/cos frequencies for each token position in one pass.
Frequency: Called during model initialization and potentially at every forward if dynamic.
"""
import math
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(positions, embed_dim, temperature=10000.0):
    """
    positions: [N] integer position indices
    embed_dim: int, must be even
    Returns: [N, embed_dim] sincos embeddings on same device as positions
    """
    N = positions.shape[0]
    assert embed_dim % 2 == 0
    half_dim = embed_dim // 2

    omega = torch.arange(half_dim, dtype=torch.float32, device=positions.device)
    omega = 1.0 / (temperature ** (omega / half_dim))

    pos = positions.float().unsqueeze(1)  # [N, 1]
    freq = pos * omega.unsqueeze(0)       # [N, half_dim]

    emb_sin = freq.sin()
    emb_cos = freq.cos()
    return torch.cat([emb_sin, emb_cos], dim=-1)


# --- KERNEL ---
@triton.jit
def _sincos_embed_kernel(
    POS_ptr,   # [N] int64 positions
    OUT_ptr,   # [N, embed_dim] output
    N, half_dim,
    temperature: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (N,)
    Each program handles one position token, writing embed_dim values.
    """
    tok_idx = tl.program_id(0)
    pos_val = tl.load(POS_ptr + tok_idx).to(tl.float32)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < half_dim

    # omega = 1 / temperature^(d / half_dim)
    omega = 1.0 / (temperature ** (offs_d.to(tl.float32) / half_dim))

    freq = pos_val * omega  # [half_dim]

    sin_vals = tl.sin(freq)
    cos_vals = tl.cos(freq)

    # Write sin to first half, cos to second half
    sin_out_off = tok_idx * half_dim * 2 + offs_d
    cos_out_off = tok_idx * half_dim * 2 + half_dim + offs_d

    tl.store(OUT_ptr + sin_out_off, sin_vals.to(tl.float16), mask=mask_d)
    tl.store(OUT_ptr + cos_out_off, cos_vals.to(tl.float16), mask=mask_d)


def kernel_fn(positions, embed_dim, temperature=10000.0):
    """
    positions: [N] integer position indices (on GPU)
    embed_dim: int, must be even
    Returns: [N, embed_dim] fp16 sincos embeddings
    """
    assert embed_dim % 2 == 0
    N = positions.shape[0]
    half_dim = embed_dim // 2
    BLOCK_D = triton.next_power_of_2(half_dim)

    out = torch.empty(N, embed_dim, dtype=torch.float16, device=positions.device)
    _sincos_embed_kernel[(N,)](
        positions.long(), out, N, half_dim,
        temperature=temperature, BLOCK_D=BLOCK_D,
        num_warps=min(8, max(1, BLOCK_D // 32)),
    )
    return out


def can_use_kernel(positions, embed_dim, temperature=10000.0):
    if not positions.is_cuda:
        return False
    if embed_dim % 2 != 0:
        return False
    if embed_dim // 2 > 2048:
        return False
    return True


SHAPES = {
    "vit_l_1d": {"positions": (1024,), "embed_dim": 1024},
    "vit_h_1d": {"positions": (2048,), "embed_dim": 1280},
    "small":    {"positions": (256,), "embed_dim": 384},
}
