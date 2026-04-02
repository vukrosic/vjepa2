"""Queue wrapper for rotate_queries_or_keys.

This family targets the RoPE single-tensor rotation path used in
`src.models.utils.modules.rotate_queries_or_keys`, but keeps the dependency
surface small by using the low-level Triton helper directly.
"""

import torch

from src.models.utils.triton_kernels import (
    can_use_triton_rope_rotate,
    triton_rotate_queries_or_keys,
    triton_rotate_queries_or_keys_autograd,
)

_INV_FREQ_CACHE = {}


def baseline_fn(x, pos):
    _, _, _, dim = x.size()
    if pos.dtype != x.dtype:
        pos = pos.to(dtype=x.dtype)
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    freq = pos.unsqueeze(-1) * omega
    emb_sin = freq.sin().squeeze(-1)
    emb_cos = freq.cos().squeeze(-1)
    emb_sin = torch.cat([emb_sin, emb_sin], dim=-1)
    emb_cos = torch.cat([emb_cos, emb_cos], dim=-1)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


def can_use_kernel(x, pos):
    if x.ndim != 4 or x.size(-1) % 2 != 0:
        return False
    if pos.device != x.device:
        return False
    if pos.ndim == 1:
        return pos.numel() == x.size(-2)
    if pos.ndim == 3:
        return pos.shape[0] == 1 and pos.shape[1] == 1 and pos.shape[2] == x.size(-2)
    return False


def kernel_fn(x, pos):
    if not can_use_kernel(x, pos):
        return baseline_fn(x, pos)

    if can_use_triton_rope_rotate(x, pos):
        key = (x.device.type, x.device.index, x.dtype, x.size(-1))
        omega = _INV_FREQ_CACHE.get(key)
        if omega is None:
            omega = torch.arange(x.size(-1) // 2, dtype=x.dtype, device=x.device)
            omega /= x.size(-1) / 2.0
            omega = 1.0 / 10000**omega
            _INV_FREQ_CACHE[key] = omega
        if torch.is_grad_enabled() and x.requires_grad:
            return triton_rotate_queries_or_keys_autograd(x, pos, omega)
        return triton_rotate_queries_or_keys(x, pos, omega)

    return baseline_fn(x, pos)


SHAPES = {
    "small": {"x": (2, 4, 128, 32), "pos": (128,)},
    "vit_l": {"x": (2, 16, 1024, 64), "pos": (1024,)},
    "vit_h": {"x": (2, 16, 2048, 80), "pos": (2048,)},
}
