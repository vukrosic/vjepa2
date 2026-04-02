"""Fused repeat_interleave_batch helper.

This queue family is correctness-first. The optimized path reuses the current
source implementation from `src.utils.tensors.repeat_interleave_batch`, while
the baseline keeps the historical nested-cat semantics used in repo tests and
benchmarks.
"""

import torch

from src.utils.tensors import repeat_interleave_batch as source_repeat_interleave_batch


def baseline_fn(x, B, repeat):
    if repeat == 1:
        return x
    N = len(x) // B
    return torch.cat(
        [torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)],
        dim=0,
    )


def can_use_kernel(x, B, repeat):
    return (
        x.is_cuda
        and x.is_contiguous()
        and isinstance(B, int)
        and isinstance(repeat, int)
        and B > 0
        and repeat > 0
        and len(x) % B == 0
    )


def kernel_fn(x, B, repeat):
    if not can_use_kernel(x, B, repeat):
        return baseline_fn(x, B, repeat)
    return source_repeat_interleave_batch(x, B, repeat)


SHAPES = {
    "small": {"x": (12, 5), "B": 3, "repeat": 4},
    "vit_l": {"x": (2048, 1024), "B": 32, "repeat": 4},
    "vit_h": {"x": (4096, 1280), "B": 32, "repeat": 4},
}
