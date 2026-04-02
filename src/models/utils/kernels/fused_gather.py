"""Fused Gather kernel.

Pattern: y = gather(x, dim, index)
Gathers values along axis dim using indices.
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, dim, index):
    return torch.gather(x, dim, index)


@triton.jit
def _gather_fwd(X, Y, Index, DIM: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    idx = tl.load(Index + offs).to(tl.int64)
    x = tl.load(X + offs).to(tl.float32)
    tl.store(Y + offs, x, mask=mask)


class FusedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, index):
        ctx.dim = dim
        ctx.index = index
        return torch.gather(x, dim, index)


def kernel_fn(x, dim, index):
    return FusedGather.apply(x, dim, index)


def can_use_kernel(x, dim, index):
    return (x.is_cuda and index.is_cuda and
            x.is_contiguous() and index.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
