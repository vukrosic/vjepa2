"""Softmax + cross entropy helper."""

import torch
import torch.nn.functional as F


def baseline_fn(logits, target_indices, temperature=0.07):
    return F.cross_entropy(logits / temperature, target_indices)


def can_use_kernel(logits, target_indices, temperature):
    return (
        logits.is_cuda
        and target_indices.is_cuda
        and logits.is_contiguous()
        and target_indices.is_contiguous()
        and logits.ndim == 2
        and target_indices.ndim == 1
        and logits.shape[0] == target_indices.shape[0]
        and logits.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and target_indices.dtype == torch.long
        and float(temperature) > 0.0
    )


def kernel_fn(logits, target_indices, temperature=0.07):
    if not can_use_kernel(logits, target_indices, temperature):
        return baseline_fn(logits, target_indices, temperature)
    return baseline_fn(logits, target_indices, temperature)


SHAPES = {
    "small": {"logits": (256, 512), "target_shape": (256,)},
    "medium": {"logits": (512, 1024), "target_shape": (512,)},
    "large": {"logits": (1024, 2048), "target_shape": (1024,)},
}
