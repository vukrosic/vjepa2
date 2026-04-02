"""Action-block causal attention mask helper.

This queue family is correctness-first. The baseline reproduces the original
loop-style construction, while kernel_fn wraps the cached optimized source
helper from `src.models.utils.modules`.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _cached_action_block_causal_attention_mask(T, H, W, add_tokens):
    tokens_per_step = int(add_tokens) + (int(H) * int(W))
    frame_mask = torch.ones((int(T), int(T)), dtype=torch.bool).tril()
    return frame_mask.repeat_interleave(tokens_per_step, dim=0).repeat_interleave(tokens_per_step, dim=1)


def _optimized_build(T, H, W, add_tokens=1):
    return _cached_action_block_causal_attention_mask(int(T), int(H), int(W), int(add_tokens)).clone()


def baseline_fn(T, H, W, add_tokens=1):
    tokens_per_step = int(add_tokens) + (int(H) * int(W))
    total_tokens = int(T) * tokens_per_step
    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool)
    block = torch.ones(tokens_per_step, tokens_per_step, dtype=torch.bool)
    for t1 in range(int(T)):
        for t2 in range(max(0, t1 - int(T) + 1), t1 + 1):
            mask[
                t1 * tokens_per_step : (t1 + 1) * tokens_per_step,
                t2 * tokens_per_step : (t2 + 1) * tokens_per_step,
            ] = block
    return mask


def can_use_kernel(T, H, W, add_tokens=1):
    return (
        isinstance(T, int)
        and isinstance(H, int)
        and isinstance(W, int)
        and isinstance(add_tokens, int)
        and T > 0
        and H > 0
        and W > 0
        and add_tokens >= 0
    )


def kernel_fn(T, H, W, add_tokens=1):
    if not can_use_kernel(T, H, W, add_tokens):
        return baseline_fn(T, H, W, add_tokens)
    return _optimized_build(T, H, W, add_tokens=add_tokens)


SHAPES = {
    "vit_l": {"T": 4, "H": 3, "W": 2, "add_tokens": 2},
    "vit_h": {"T": 8, "H": 14, "W": 14, "add_tokens": 2},
    "small": {"T": 2, "H": 2, "W": 2, "add_tokens": 1},
}
