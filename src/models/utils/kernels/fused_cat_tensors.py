"""Fused Cat Tensors kernel.

Pattern: y = cat([a, b], dim)
Concatenates two tensors along a dimension.
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(a, b, dim=0):
    return torch.cat([a, b], dim=dim)


@triton.jit
def _cat_tensors_fwd(A, B, Y, N_A: tl.constexpr, N_B: tl.constexpr, OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask_a = offs < N_A
    mask_b = offs < N_B
    for i in range(BLOCK):
        off = pid * BLOCK + i
        if off < N_A:
            v = tl.load(A + off).to(tl.float32)
            tl.store(Y + off, v)
        elif off < N_A + N_B:
            v = tl.load(B + off - N_A).to(tl.float32)
            tl.store(Y + off, v)


class FusedCatTensors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, dim=0):
        ctx.dim = dim
        y = torch.cat([a, b], dim=dim)
        return y


def kernel_fn(a, b, dim=0):
    return FusedCatTensors.apply(a, b, dim)


def can_use_kernel(a, b, dim=0):
    return (a.is_cuda and b.is_cuda and
            a.is_contiguous() and b.is_contiguous() and
            a.dtype == b.dtype)


SHAPES = {
    "vit_l":  {"a": (2, 512, 4096), "b": (2, 512, 4096)},
    "vit_h":  {"a": (2, 1024, 5120), "b": (2, 1024, 5120)},
    "small":  {"a": (8, 128, 1536), "b": (8, 128, 1536)},
}
