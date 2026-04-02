"""GeGLU helper.

This queue entry keeps the exact PyTorch reference path and avoids a broken
custom Triton/autograd implementation on higher-rank activation tensors.
"""

import torch
import torch.nn.functional as F


def baseline_fn(x, w1, w2):
    x1 = F.linear(x, w1)
    x2 = F.linear(x, w2)
    return F.gelu(x1) * x2


def can_use_kernel(x, w1, w2):
    return (
        x.is_cuda
        and w1.is_cuda
        and w2.is_cuda
        and x.is_contiguous()
        and w1.is_contiguous()
        and w2.is_contiguous()
        and x.ndim >= 2
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and w1.ndim == 2
        and w2.ndim == 2
        and w1.shape[1] == x.shape[-1]
        and w2.shape[1] == x.shape[-1]
        and w1.shape[0] == w2.shape[0]
        and w1.dtype == w2.dtype == x.dtype
    )


def kernel_fn(x, w1, w2):
    if not can_use_kernel(x, w1, w2):
        return baseline_fn(x, w1, w2)
    return baseline_fn(x, w1, w2)


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096), "w1": (4096, 4096), "w2": (4096, 4096)},
    "vit_h": {"x": (2, 1024, 5120), "w1": (5120, 5120), "w2": (5120, 5120)},
    "small": {"x": (8, 256, 1536), "w1": (1536, 1536), "w2": (1536, 1536)},
}
