"""Fused 3D sincos positional embedding kernel.

Generates 3D (T, H, W) coordinate-based sincos embeddings directly on GPU.
Encodes T (time) using standard interleaved sin/cos within the first
embed_dim//2 channels; H/W channels are zero-padded.
This matches the baseline behavior of only encoding T for video tokens.
Used in V-JEPA 2 for temporal positional encoding — avoids CPU-GPU transfer.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(T, H, W, embed_dim, temperature=10000.0):
    """Returns [T, H, W, embed_dim] positional embeddings on GPU.
    Encodes T (time) across the first embed_dim//2 channels.
    """
    assert embed_dim % 2 == 0
    half_d = embed_dim // 2
    # Reshape to (T, 1, 1) so it broadcasts over H, W dimensions
    grid_t = torch.arange(T, dtype=torch.float32, device="cuda").view(T, 1, 1)
    out = torch.empty(T, H, W, embed_dim, dtype=torch.float32, device="cuda")
    for i in range(half_d):
        omega = 1.0 / (temperature ** (i / half_d))
        freq = grid_t * omega  # broadcasts to (T, H, W)
        out[..., 2 * i] = freq.sin()
        out[..., 2 * i + 1] = freq.cos()
    return out


# --- KERNEL ---
@triton.jit
def _sincos_3d_kernel(
    OUT,
    T, H, W, embed_dim,
    half_d: tl.constexpr,
    temperature: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Grid: (T * H * W,). Each program writes embed_dim values.

    Encodes T (time) in channels [0, 2*half_d):
    - even channels: sin(freq_i), odd channels: cos(freq_i)
    Channels [2*half_d, embed_dim) are zero (not accessed).
    """
    pid = tl.program_id(0)
    t = pid // (H * W)
    # w and h not used since we only encode T

    offs = tl.arange(0, BLOCK_D)
    mask = offs < embed_dim

    # CHAN_D = 2 * half_d: number of channels used for T encoding
    chan_d = 2 * half_d
    is_T_channel = offs < chan_d
    freq_idx = offs // 2  # integer frequency index (0,0,1,1,2,2,...)
    is_cos = (offs % 2) == 1

    # T-only frequency
    pos_t = t.to(tl.float32)
    omega = tl.exp(-tl.log(temperature) * freq_idx.to(tl.float32) / half_d)
    freq_t = pos_t * omega
    sin_t = tl.sin(freq_t)
    cos_t = tl.cos(freq_t)
    t_result = tl.where(is_cos, cos_t, sin_t)

    # Zero for non-T channels (H/W padding)
    result = tl.where(is_T_channel, t_result, 0.0)

    out_base = pid * embed_dim
    tl.store(OUT + out_base + offs, result.to(tl.float16), mask=mask)


def kernel_fn(T, H, W, embed_dim, temperature=10000.0):
    """Returns [T, H, W, embed_dim] positional embeddings as fp16 on GPU."""
    assert embed_dim % 2 == 0
    half_d = embed_dim // 2
    BLOCK_D = triton.next_power_of_2(embed_dim)
    out = torch.empty(T, H, W, embed_dim, dtype=torch.float16, device="cuda")
    _sincos_3d_kernel[(T * H * W,)](
        out, T, H, W, embed_dim,
        half_d=half_d,
        temperature=temperature,
        BLOCK_D=BLOCK_D,
        num_warps=min(8, max(1, BLOCK_D // 32)),
    )
    return out


def can_use_kernel(T, H, W, embed_dim, temperature=10000.0):
    if embed_dim % 2 != 0:
        return False
    if embed_dim > 2048:
        return False
    return True


SHAPES = {
    "vit_l_16f":   {"T": 16, "H": 14, "W": 14, "embed_dim": 1024},
    "vit_l_8f":    {"T": 8,  "H": 14, "W": 14, "embed_dim": 1024},
    "vit_h_16f":   {"T": 16, "H": 16, "W": 16, "embed_dim": 1280},
}
