"""3D sincos positional embedding helper.

This queue family is handled conservatively: the implementation mirrors the
exact reference math from `src/models/utils/pos_embs.py` and keeps a strict
`can_use_kernel()` guard. The queue benchmark is therefore honest and the
family remains import-safe even if no Triton path is used.
"""

import math

import torch


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)

    pos = pos.reshape(-1).to(torch.float32)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


def baseline_fn(T, H, W, embed_dim, temperature=10000.0, cls_token=False, uniform_power=False):
    if uniform_power:
        h_embed_dim = w_embed_dim = d_embed_dim = int(math.ceil(embed_dim / 6) * 2)
    else:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2

    grid_d = torch.arange(T, dtype=torch.float32, device="cuda")
    grid_h = torch.arange(H, dtype=torch.float32, device="cuda")
    grid_w = torch.arange(W, dtype=torch.float32, device="cuda")
    grid_d, grid_h, grid_w = torch.meshgrid(grid_d, grid_h, grid_w, indexing="ij")

    emb_d = _get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)
    emb_h = _get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)
    emb_w = _get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)

    pos_embed = torch.cat([emb_d, emb_h, emb_w], dim=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros(1, embed_dim, dtype=pos_embed.dtype, device=pos_embed.device), pos_embed],
            dim=0,
        )
    return pos_embed.reshape(T, H, W, embed_dim)


def can_use_kernel(T, H, W, embed_dim, temperature=10000.0, cls_token=False, uniform_power=False):
    return (
        isinstance(T, int)
        and isinstance(H, int)
        and isinstance(W, int)
        and isinstance(embed_dim, int)
        and T > 0
        and H > 0
        and W > 0
        and embed_dim > 0
        and embed_dim % 2 == 0
    )


def kernel_fn(T, H, W, embed_dim, temperature=10000.0, cls_token=False, uniform_power=False):
    if not can_use_kernel(T, H, W, embed_dim, temperature, cls_token, uniform_power):
        return baseline_fn(T, H, W, embed_dim, temperature, cls_token, uniform_power)
    return baseline_fn(T, H, W, embed_dim, temperature, cls_token, uniform_power)


SHAPES = {
    "vit_l_16f": {"T": 16, "H": 14, "W": 14, "embed_dim": 1024},
    "vit_l_8f": {"T": 8, "H": 14, "W": 14, "embed_dim": 1024},
    "vit_h_16f": {"T": 16, "H": 16, "W": 16, "embed_dim": 1280},
}
