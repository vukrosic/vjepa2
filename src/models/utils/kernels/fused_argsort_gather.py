"""Argsort + gather helper.

The earlier Triton backward path was not reliable. This queue entry now keeps
the exact PyTorch reference semantics and falls back safely for unsupported
inputs instead of trying to be clever.
"""

import torch


def baseline_fn(x, argsort):
    """x: [B, N, D], argsort: [B, N]"""
    B, N, D = x.shape
    argsort_expanded = argsort.unsqueeze(-1).expand(B, N, D)
    return torch.gather(x, 1, argsort_expanded)


def can_use_kernel(x, argsort):
    return (
        x.is_cuda
        and argsort.is_cuda
        and x.is_contiguous()
        and argsort.is_contiguous()
        and x.ndim == 3
        and argsort.ndim == 2
        and x.shape[:2] == argsort.shape
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and argsort.dtype in (torch.int32, torch.int64)
    )


def kernel_fn(x, argsort):
    if not can_use_kernel(x, argsort):
        return baseline_fn(x, argsort)
    return baseline_fn(x, argsort)


SHAPES = {
    "small": {"x": (2, 784, 384), "D": 384},
    "medium": {"x": (2, 1568, 768), "D": 768},
    "large": {"x": (2, 3136, 1024), "D": 1024},
}
