"""Queue wrapper for the source apply_masks helper.

This family compares the current optimized implementation in `src.masks.utils`
against the historical loop-based gather baseline used in repo benchmarks.
"""

import torch

from src.masks.utils import apply_masks as optimized_apply_masks


def baseline_fn(x, masks, concat=True):
    if not masks:
        if not concat:
            return []
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")

    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1)
        if m.ndim == 1:
            mask_keep = mask_keep.unsqueeze(0)
        mask_keep = mask_keep.expand(*mask_keep.shape[:-1], x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def can_use_kernel(x, masks, concat=True):
    if not x.is_cuda or x.ndim != 3 or not masks:
        return False
    if not all(m.is_cuda and m.dtype == torch.long for m in masks):
        return False
    if not all(m.ndim in (1, 2) for m in masks):
        return False
    return True


def kernel_fn(x, masks, concat=True):
    if not can_use_kernel(x, masks, concat=concat):
        return baseline_fn(x, masks, concat=concat)
    return optimized_apply_masks(x, masks, concat=concat)


SHAPES = {
    "multi1d": {"x": (32, 1024, 384), "mask_count": 4, "kind": "1d", "mask_len": 256},
    "multi2d": {"x": (32, 1024, 384), "mask_count": 4, "kind": "2d", "mask_shape": (32, 256)},
    "large2d": {"x": (16, 2048, 512), "mask_count": 4, "kind": "2d", "mask_shape": (16, 512)},
}
