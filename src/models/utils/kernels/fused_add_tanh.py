"""Fused add_tanh kernel.

Pattern: y = tanh(x1 + x2)
Fuses: add + tanh into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x1, x2):
    return torch.tanh(x1 + x2)


@triton.jit
def _add_tanh_fwd(X1, X2, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(X1 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(X2 + offs, mask=mask, other=0.0).to(tl.float32)
    s = a + b
    y = tl.tanh(s)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _add_tanh_bwd(X1, X2, DY, DX1, DX2, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(X1 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(X2 + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    s = a + b
    tanh_s = tl.tanh(s)
    ds = dy * (1.0 - tanh_s * tanh_s)
    tl.store(DX1 + offs, ds, mask=mask)
    tl.store(DX2 + offs, ds, mask=mask)


class FusedAddTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        y = torch.empty_like(x1)
        N = x1.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _add_tanh_fwd[(n_blocks,)](x1, x2, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x1, x2)
        return y

    @staticmethod
    def backward(ctx, dy):
        x1, x2 = ctx.saved_tensors
        dx1 = torch.empty_like(x1)
        dx2 = torch.empty_like(x2)
        N = dy.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _add_tanh_bwd[(n_blocks,)](x1, x2, dy, dx1, dx2, N, BLOCK=BLOCK, num_warps=4)
        return dx1, dx2


def kernel_fn(x1, x2):
    return FusedAddTanh.apply(x1, x2)


def can_use_kernel(x1, x2):
    return (x1.is_cuda and x1.is_contiguous() and x2.is_contiguous() and
            x1.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x1": (2, 1024, 4096), "x2": (2, 1024, 4096)},
    "vit_h":  {"x1": (2, 2048, 5120), "x2": (2, 2048, 5120)},
    "small":  {"x1": (8, 256, 1536), "x2": (8, 256, 1536)},
}
