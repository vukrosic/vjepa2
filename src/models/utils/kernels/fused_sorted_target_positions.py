"""Fused sorted target position helper.

This queue family wraps the exact source helper used by the predictor while
benchmarking it against a simpler row-by-row historical baseline.
"""

import torch

from src.models.predictor import _get_sorted_target_positions as source_sorted_target_positions


def baseline_fn(masks_x, masks_y):
    if masks_x.ndim == 1 and masks_y.ndim == 1:
        positions = torch.searchsorted(masks_x.contiguous(), masks_y.contiguous(), right=True)
        offsets = torch.arange(masks_y.numel(), device=masks_y.device, dtype=positions.dtype)
        return positions + offsets

    rows = []
    for mask_x, mask_y in zip(masks_x, masks_y):
        positions = torch.searchsorted(mask_x.contiguous(), mask_y.contiguous(), right=True)
        offsets = torch.arange(mask_y.numel(), device=mask_y.device, dtype=positions.dtype)
        rows.append(positions + offsets)
    return torch.stack(rows, dim=0)


def can_use_kernel(masks_x, masks_y):
    return (
        masks_x.is_cuda
        and masks_y.is_cuda
        and masks_x.ndim == 2
        and masks_y.ndim == 2
        and masks_x.shape[0] == masks_y.shape[0]
        and masks_x.dtype in (torch.int32, torch.int64)
        and masks_y.dtype == masks_x.dtype
    )


def kernel_fn(masks_x, masks_y):
    if not can_use_kernel(masks_x, masks_y):
        return baseline_fn(masks_x, masks_y)
    return source_sorted_target_positions(masks_x, masks_y)


SHAPES = {
    "small": {"masks_x": (8, 196), "masks_y": (8, 49), "max_index": 784},
    "vit_l": {"masks_x": (32, 256), "masks_y": (32, 64), "max_index": 1024},
    "vit_h": {"masks_x": (32, 1024), "masks_y": (32, 256), "max_index": 4096},
}
