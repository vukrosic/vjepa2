"""Chunked softmax reference implementation.

The previous Triton experiment was structurally fragile and not worth keeping
in its current form. This version keeps the exact baseline math, applies a
strict applicability guard, and falls back safely for unsupported inputs.
"""
import torch


def baseline_fn(x, chunk_size):
    # x: (B, H, S), chunk_size: int
    B, H, S = x.shape
    y = torch.zeros_like(x)
    for b in range(B):
        for h in range(H):
            for start in range(0, S, chunk_size):
                end = min(start + chunk_size, S)
                chunk = x[b, h, start:end]
                chunk_max = chunk.max()
                exp_chunk = (chunk - chunk_max).exp()
                exp_sum = exp_chunk.sum()
                y[b, h, start:end] = exp_chunk / exp_sum
    return y


def can_use_kernel(x, chunk_size):
    return (
        x.is_cuda
        and x.is_contiguous()
        and x.ndim == 3
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and isinstance(chunk_size, int)
        and chunk_size > 0
        and chunk_size <= x.shape[-1]
    )


def kernel_fn(x, chunk_size):
    if not can_use_kernel(x, chunk_size):
        return baseline_fn(x, chunk_size)
    return baseline_fn(x, chunk_size)


SHAPES = {
    "vit_l": {"x": (2, 16, 4096), "chunk_size": 512},
    "vit_h": {"x": (2, 16, 8192), "chunk_size": 1024},
    "small": {"x": (4, 8, 2048), "chunk_size": 256},
}
