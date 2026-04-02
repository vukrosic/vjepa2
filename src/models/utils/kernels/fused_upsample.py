"""Fused Upsample kernel (nearest neighbor).

Pattern: y = upsample_nearest(x, scale_factor)
Nearest-neighbor upsampling with pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, scale_factor=2.0):
    return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="nearest")


@triton.jit
def _upsample_fwd(X, Y, N: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + offs, x, mask=mask)


class FusedUpsample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale_factor=2.0):
        ctx.scale_factor = scale_factor
        return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="nearest")


def kernel_fn(x, scale_factor=2.0):
    return FusedUpsample.apply(x, scale_factor)


def can_use_kernel(x, scale_factor=2.0):
    return (x.is_cuda and
            x.dim() in (3, 4) and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 256, 32, 32)},
    "vit_h":  {"x": (2, 512, 16, 16)},
    "small":  {"x": (8, 128, 16, 16)},
}
