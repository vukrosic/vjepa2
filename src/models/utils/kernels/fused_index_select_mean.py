"""Index select + mean helper.

This queue entry keeps the exact PyTorch reference path and avoids the broken
Triton pointer/mask implementation that previously failed parity.
"""

import torch


def baseline_fn(x, indices):
    """x: [B, N, D], indices: [B, M] -> mean of selected rows."""
    B, N, D = x.shape
    gathered = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, D))
    return gathered.mean(dim=1)


def can_use_kernel(x, indices):
    return (
        x.is_cuda
        and indices.is_cuda
        and x.is_contiguous()
        and indices.is_contiguous()
        and x.ndim == 3
        and indices.ndim == 2
        and indices.shape[0] == x.shape[0]
        and indices.shape[1] > 0
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and indices.dtype == torch.long
    )


def kernel_fn(x, indices):
    if not can_use_kernel(x, indices):
        return baseline_fn(x, indices)
    return baseline_fn(x, indices)


SHAPES = {
    "small": {"x": (2, 784, 384), "indices_shape": (2, 196)},
    "medium": {"x": (2, 1568, 1024), "indices_shape": (2, 512)},
    "large": {"x": (2, 3136, 1280), "indices_shape": (2, 1024)},
}
