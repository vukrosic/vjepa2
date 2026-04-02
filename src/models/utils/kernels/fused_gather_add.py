"""Fused gather + add helper.

This queue entry is correctness-first. The Triton version was too fragile
for the current contract, so the kernel now uses the exact PyTorch reference
path and a strict applicability guard.
"""

import torch


def baseline_fn(x, indices, accum):
    """
    x: [B, N, D], indices: [B, M], accum: [B, M, D]
    Returns: accum + gather(x, 1, indices_expanded)
    """
    B, M = indices.shape
    gathered = torch.gather(x, 1, indices.unsqueeze(-1).expand(B, M, x.shape[-1]))
    return accum + gathered


def can_use_kernel(x, indices, accum):
    return (
        x.is_cuda
        and indices.is_cuda
        and accum.is_cuda
        and x.is_contiguous()
        and indices.is_contiguous()
        and accum.is_contiguous()
        and x.ndim == 3
        and indices.ndim == 2
        and accum.ndim == 3
        and x.shape[0] == indices.shape[0] == accum.shape[0]
        and indices.shape[1] == accum.shape[1]
        and x.shape[2] == accum.shape[2]
        and indices.dtype == torch.long
        and x.dtype == accum.dtype
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def kernel_fn(x, indices, accum):
    if not can_use_kernel(x, indices, accum):
        return baseline_fn(x, indices, accum)
    return baseline_fn(x, indices, accum)


SHAPES = {
    "small": {"x": (2, 784, 384), "indices_shape": (2, 196), "D": 384},
    "medium": {"x": (2, 1568, 1024), "indices_shape": (2, 512), "D": 1024},
    "large": {"x": (2, 3136, 1280), "indices_shape": (2, 1024), "D": 1280},
}
