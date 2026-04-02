"""Percentile clip helper."""

import torch


def baseline_fn(x, low, high):
    return torch.clamp(x, low.view(1, 1, -1), high.view(1, 1, -1))


def can_use_kernel(x, low, high):
    return (
        x.is_cuda
        and low.is_cuda
        and high.is_cuda
        and x.is_contiguous()
        and low.is_contiguous()
        and high.is_contiguous()
        and x.ndim == 3
        and low.ndim == 1
        and high.ndim == 1
        and x.shape[-1] == low.shape[0] == high.shape[0]
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and x.dtype == low.dtype == high.dtype
    )


def kernel_fn(x, low, high):
    if not can_use_kernel(x, low, high):
        return baseline_fn(x, low, high)
    return baseline_fn(x, low, high)


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "C": 1024},
    "vit_h": {"x": (2, 2048, 1280), "C": 1280},
    "small": {"x": (4, 256, 384), "C": 384},
}
