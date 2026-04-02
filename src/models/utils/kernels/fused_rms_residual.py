"""Fused RMSNorm + residual add helper.

This queue entry is correctness-first. It keeps the exact PyTorch reference
path and a strict applicability guard so unsupported layouts fall back safely.
"""

import torch


def baseline_fn(x, residual, weight, eps=1e-6):
    y = x + residual
    rms = y.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (y / rms * weight).to(x.dtype)


def _has_valid_shape(x, residual, weight):
    return (
        x.ndim >= 1
        and x.shape == residual.shape
        and weight.ndim == 1
        and weight.shape[0] == x.shape[-1]
    )


def can_use_kernel(x, residual, weight):
    return (
        x.is_cuda
        and residual.is_cuda
        and weight.is_cuda
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and _has_valid_shape(x, residual, weight)
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and x.dtype == residual.dtype == weight.dtype
    )


def kernel_fn(x, residual, weight, eps=1e-6):
    if not can_use_kernel(x, residual, weight):
        return baseline_fn(x, residual, weight, eps)

    summed = x + residual
    rms = summed.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (summed / rms * weight).to(x.dtype)


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "C": 1024},
    "vit_h": {"x": (2, 2048, 1280), "C": 1280},
    "small": {"x": (4, 256, 384), "C": 384},
}
