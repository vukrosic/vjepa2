"""Elementwise minimum helper."""

import torch


def baseline_fn(a, b):
    return torch.minimum(a, b)


def can_use_kernel(a, b):
    return (
        a.is_cuda
        and b.is_cuda
        and a.is_contiguous()
        and b.is_contiguous()
        and a.shape == b.shape
        and a.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and a.dtype == b.dtype
    )


def kernel_fn(a, b):
    if not can_use_kernel(a, b):
        return baseline_fn(a, b)
    return baseline_fn(a, b)


SHAPES = {
    "vit_l": {"a": (2, 1024, 1024), "b": (2, 1024, 1024)},
    "vit_h": {"a": (2, 2048, 1280), "b": (2, 2048, 1280)},
    "small": {"a": (8, 256, 384), "b": (8, 256, 384)},
}
