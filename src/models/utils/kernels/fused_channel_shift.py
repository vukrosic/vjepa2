"""Channel shift helper.

This queue entry is correctness-first. It keeps the exact reference channel
rotation and uses a strict guard so unsupported inputs fall back safely.
"""

import torch


def baseline_fn(x, shift):
    y = torch.empty_like(x)
    channels = x.shape[-1]
    for c in range(channels):
        src_c = (c + shift) % channels
        y[..., c] = x[..., src_c]
    return y


def can_use_kernel(x, shift):
    return (
        x.is_cuda
        and x.is_contiguous()
        and x.ndim >= 1
        and x.shape[-1] > 0
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and isinstance(shift, int)
        and 0 <= shift < x.shape[-1]
    )


def kernel_fn(x, shift):
    if not can_use_kernel(x, shift):
        return baseline_fn(x, shift)
    return baseline_fn(x, shift)


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "shift": 128},
    "vit_h": {"x": (2, 2048, 1280), "shift": 160},
    "small": {"x": (4, 256, 384), "shift": 48},
}
