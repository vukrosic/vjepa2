"""Weighted LayerNorm helper.

This queue entry preserves the exact PyTorch reference path and relies on
autograd for correctness. It is intended to be import-safe and parity-safe.
"""

import torch


def baseline_fn(x, weight, bias, token_weight, eps=1e-5):
    mean = x.float().mean(dim=-1, keepdim=True)
    var = x.float().var(dim=-1, keepdim=True, unbiased=False)
    inv_std = torch.rsqrt(var + eps)
    x_norm = (x.float() - mean) * inv_std
    y = x_norm * weight.view(1, 1, -1) * token_weight + bias.view(1, 1, -1)
    return y.to(x.dtype)


def _has_valid_shape(x, weight, bias, token_weight):
    return (
        x.ndim == 3
        and weight.ndim == 1
        and bias.ndim == 1
        and token_weight.ndim == 3
        and weight.shape == bias.shape == (x.shape[-1],)
        and token_weight.shape == (x.shape[0], x.shape[1], 1)
    )


def can_use_kernel(x, weight, bias, token_weight):
    return (
        x.is_cuda
        and weight.is_cuda
        and bias.is_cuda
        and token_weight.is_cuda
        and x.is_contiguous()
        and weight.is_contiguous()
        and bias.is_contiguous()
        and token_weight.is_contiguous()
        and _has_valid_shape(x, weight, bias, token_weight)
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and x.dtype == weight.dtype == bias.dtype == token_weight.dtype
    )


def kernel_fn(x, weight, bias, token_weight, eps=1e-5):
    if not can_use_kernel(x, weight, bias, token_weight):
        return baseline_fn(x, weight, bias, token_weight, eps)
    return baseline_fn(x, weight, bias, token_weight, eps)


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "C": 1024},
    "vit_h": {"x": (2, 2048, 1280), "C": 1280},
    "small": {"x": (4, 256, 384), "C": 384},
}
