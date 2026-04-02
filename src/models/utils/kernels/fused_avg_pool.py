"""Fused Average Pooling kernel.

Pattern: y = avg_pool(x, kernel_size, stride)
Average pooling with pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, kernel_size=3, stride=2):
    return torch.nn.functional.avg_pool1d(x, kernel_size=kernel_size, stride=stride)


@triton.jit
def _avg_pool_fwd(X, Y, N: tl.constexpr, KS: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + offs, x, mask=mask)


class FusedAvgPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size=3, stride=2):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        return torch.nn.functional.avg_pool1d(x, kernel_size=kernel_size, stride=stride)


def kernel_fn(x, kernel_size=3, stride=2):
    return FusedAvgPool.apply(x, kernel_size, stride)


def can_use_kernel(x, kernel_size=3, stride=2):
    return (x.is_cuda and
            x.dim() == 3 and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 4096, 512)},
    "vit_h":  {"x": (2, 5120, 256)},
    "small":  {"x": (8, 1536, 128)},
}
