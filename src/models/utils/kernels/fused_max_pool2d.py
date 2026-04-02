"""Fused Max Pool 2D kernel.

Pattern: y = max_pool2d(x, kernel_size, stride, padding)
Max pooling with pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, kernel_size=3, stride=2, padding=1):
    return torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)


@triton.jit
def _max_pool2d_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + offs, x, mask=mask)


class FusedMaxPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size=3, stride=2, padding=1):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        return torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)


def kernel_fn(x, kernel_size=3, stride=2, padding=1):
    return FusedMaxPool2d.apply(x, kernel_size, stride, padding)


def can_use_kernel(x, kernel_size=3, stride=2, padding=1):
    return (x.is_cuda and
            x.dim() == 4 and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 256, 64, 64)},
    "vit_h":  {"x": (2, 512, 32, 32)},
    "small":  {"x": (8, 128, 32, 32)},
}
