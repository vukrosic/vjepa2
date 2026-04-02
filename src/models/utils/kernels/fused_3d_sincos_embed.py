"""Fused 3D sincos positional embedding kernel.

Generates 3D (T, H, W) coordinate-based sincos embeddings directly on GPU.
V-JEPA 2 uses 3D positional encoding for video tokens — avoids CPU-GPU transfer.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(T, H, W, embed_dim, temperature=10000.0):
    """Returns [T, H, W, embed_dim] positional embeddings on GPU."""
    assert embed_dim % 6 == 0
    half_d = embed_dim // 6
    grid_t = torch.arange(T, dtype=torch.float32, device="cuda")
    grid_h = torch.arange(H, dtype=torch.float32, device="cuda")
    grid_w = torch.arange(W, dtype=torch.float32, device="cuda")
    # Compute full 3D coordinate grid
    t_grid = grid_t.view(T, 1, 1)
    h_grid = grid_h.view(1, H, 1)
    w_grid = grid_w.view(1, 1, W)
    out = torch.empty(T, H, W, embed_dim, dtype=torch.float32, device="cuda")
    d = 0
    for dim_frac in [t_grid, h_grid, w_grid]:
        for i in range(half_d):
            omega = 1.0 / (temperature ** (i / half_d))
            out[..., d] = (dim_frac * omega).sin()
            out[..., d + 1] = (dim_frac * omega).cos()
            d += 2
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
    """Grid: (T * H * W,). Each program writes embed_dim values."""
    pid = tl.program_id(0)
    t = pid // (H * W)
    tmp = pid % (H * W)
    h = tmp // W
    w = tmp % W

    offs = tl.arange(0, BLOCK_D)
    mask = offs < embed_dim

    pos_t = t.to(tl.float32)
    pos_h = h.to(tl.float32)
    pos_w = w.to(tl.float32)

    # Channel layout: [T-sin, T-cos, H-sin, H-cos, W-sin, W-cos] x half_d each
    # Channel = offs // half_d (dimension index 0-5)
    # Channel offset = offs % half_d (frequency index)
    dim_idx = offs // half_d
    freq_idx = offs - dim_idx * half_d

    omega = 1.0 / (temperature ** (freq_idx.to(tl.float32) / half_d))

    pos = (tl.where(dim_idx == 0, pos_t, 0.0) +
           tl.where(dim_idx == 2, pos_h, 0.0) +
           tl.where(dim_idx == 4, pos_w, 0.0))

    vals = pos * omega
    sin_v = tl.sin(vals)
    cos_v = tl.cos(vals)

    is_sin = (dim_idx % 2) == 0
    result = tl.where(is_sin, sin_v, cos_v)

    out_base = (t * H * W + h * W + w) * embed_dim
    tl.store(OUT + out_base + offs, result.to(tl.float16), mask=mask)


def kernel_fn(T, H, W, embed_dim, temperature=10000.0):
    """Returns [T, H, W, embed_dim] positional embeddings as fp16 on GPU."""
    assert embed_dim % 6 == 0
    half_d = embed_dim // 6
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
    if embed_dim % 6 != 0:
        return False
    if embed_dim > 2048:
        return False
    return True


SHAPES = {
    "vit_l_16f":   {"T": 16, "H": 14, "W": 14, "embed_dim": 1024},
    "vit_l_8f":    {"T": 8,  "H": 14, "W": 14, "embed_dim": 1024},
    "vit_h_16f":   {"T": 16, "H": 16, "W": 16, "embed_dim": 1280},
}
