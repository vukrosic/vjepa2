"""Fused flip kernel.

Pattern: torch.flip(x, dims=[-1])
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.flip(x, dims=[-1])


@triton.jit
def _fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_base = row * N
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    idx = N - 1 - offs
    v_flip = tl.load(X + row_base + idx, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row_base + offs, v_flip, mask=mask)


@triton.jit
def _bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_base = row * N
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    idx = N - 1 - offs
    v_bwd = tl.load(DY + row_base + idx, mask=mask, other=0.0).to(tl.float32)
    tl.store(DX + row_base + offs, v_bwd, mask=mask)


class Flip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        BLOCK = 1024
        grid = (x.shape[0],)
        _fwd[grid](x, y, x.shape[-1], BLOCK=BLOCK, num_warps=4)

        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        grid = (x.shape[0],)
        _bwd[grid](x, dy, dx, x.shape[-1], BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return Flip.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
