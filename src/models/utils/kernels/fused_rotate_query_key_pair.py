"""Queue wrapper for rotate_query_key_pair.

This family targets the RoPE query/key pair rotation path used by
`src.models.utils.modules.rotate_query_key_pair`, but avoids importing the full
module graph so it stays import-clean in lightweight environments.
"""

import torch

from src.models.utils.triton_kernels import (
    can_use_triton_rope_rotate,
    triton_rotate_query_key_pair,
    triton_rotate_query_key_pair_autograd,
)

_INV_FREQ_CACHE = {}


def _baseline_rotate_queries_or_keys(x, pos):
    _, _, _, dim = x.size()
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    freq = pos.unsqueeze(-1) * omega
    emb_sin = freq.sin().squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = freq.cos().squeeze(-1).repeat(1, 1, 1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


def baseline_fn(q, k, pos):
    return _baseline_rotate_queries_or_keys(q, pos), _baseline_rotate_queries_or_keys(k, pos)


def can_use_kernel(q, k, pos):
    if q.ndim != 4 or k.ndim != 4:
        return False
    if q.shape != k.shape:
        return False
    if q.dtype != k.dtype or q.device != k.device:
        return False
    if q.size(-1) % 2 != 0:
        return False
    if pos.device != q.device:
        return False
    if pos.ndim == 1:
        return pos.numel() == q.size(-2)
    if pos.ndim == 3:
        return pos.shape[0] == 1 and pos.shape[1] == 1 and pos.shape[2] == q.size(-2)
    return False


def _optimized_rotate_query_key_pair(q, k, pos):
    if can_use_triton_rope_rotate(q, pos) and can_use_triton_rope_rotate(k, pos):
        key = (q.device.type, q.device.index, q.dtype, q.size(-1))
        omega = _INV_FREQ_CACHE.get(key)
        if omega is None:
            omega = torch.arange(q.size(-1) // 2, dtype=q.dtype, device=q.device)
            omega /= q.size(-1) / 2.0
            omega = 1.0 / 10000**omega
            _INV_FREQ_CACHE[key] = omega
        if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad):
            return triton_rotate_query_key_pair_autograd(q, k, pos, omega)
        return triton_rotate_query_key_pair(q, k, pos, omega)
    return baseline_fn(q, k, pos)


def kernel_fn(q, k, pos):
    if not can_use_kernel(q, k, pos):
        return baseline_fn(q, k, pos)
    return _optimized_rotate_query_key_pair(q, k, pos)


SHAPES = {
    "small": {"q": (2, 4, 128, 32), "k": (2, 4, 128, 32), "pos": (1, 1, 128)},
    "vit_l": {"q": (2, 16, 1024, 64), "k": (2, 16, 1024, 64), "pos": (1, 1, 1024)},
    "vit_h": {"q": (2, 16, 2048, 80), "k": (2, 16, 2048, 80), "pos": (1, 1, 2048)},
}
