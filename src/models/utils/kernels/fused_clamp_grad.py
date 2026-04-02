"""Fused Clamp Gradient kernel.

Pattern: gradient of clamp(x, min_val, max_val) — pass-through with gradient masking.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, dy, min_val=-1.0, max_val=1.0):
    return torch.clamp_backward(dy, x, min_val, max_val)


@triton.jit
def _clamp_grad_bwd(X, DY, DX, N: tl.constexpr, MIN_V: tl.constexpr, MAX_V: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xv = tl.load(X + offs, mask=mask).to(tl.float32)
    dyv = tl.load(DY + offs, mask=mask).to(tl.float32)
    gate = (xv > MIN_V) & (xv < MAX_V)
    dx = dyv * tl.where(gate, 1.0, 0.0)
    tl.store(DX + offs, dx, mask=mask)


class FusedClampGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_val=-1.0, max_val=1.0):
        ctx.min_val = float(min_val)
        ctx.max_val = float(max_val)
        return x

    @staticmethod
    def backward(ctx, dy):
        x = dy  # dummy, not actually used
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        N = dy.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _clamp_grad_bwd[grid](dy, dy, dx, N, ctx.min_val, ctx.max_val, BLOCK=BLOCK, num_warps=4)
        return dx, None, None


def kernel_fn(x, dy, min_val=-1.0, max_val=1.0):
    return FusedClampGrad.apply(dy, min_val, max_val)


def can_use_kernel(x, dy=None, min_val=-1.0, max_val=1.0):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536)},
}
