"""Fused sigmoid_tanh kernel.

Pattern: torch.nn.functional.sigmoid(x) * torch.nn.functional.tanh(x)
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.sigmoid(x) * torch.nn.functional.tanh(x)


@triton.jit
def _fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-v, 20.0)))
    e2v = tl.exp(tl.minimum(2.0 * v, 40.0))
    tanh_v = (e2v - 1.0) / (e2v + 1.0)
    y = sig * tanh_v
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-v, 20.0)))
    dsig = sig * (1.0 - sig)
    e2v = tl.exp(tl.minimum(2.0 * v, 40.0))
    tanh_v = (e2v - 1.0) / (e2v + 1.0)
    dtanh = 1.0 - tanh_v * tanh_v
    dx = dy * (dsig * tanh_v + sig * dtanh)
    tl.store(DX + offs, dx, mask=mask)


class SigmoidTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(sig, v)
        return y

    @staticmethod
    def backward(ctx, dy):
        sig, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(sig)
        N = sig.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _bwd[grid](sig, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return SigmoidTanh.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
