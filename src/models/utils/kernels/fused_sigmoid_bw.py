"""Fused Sigmoid backward-only kernel (gradient).

Pattern: dx = dy * sigmoid(x) * (1 - sigmoid(x))
Fuses: sigmoid gradient into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, dy):
    sigmoid = torch.sigmoid(x)
    return dy * sigmoid * (1 - sigmoid)


# --- KERNEL ---
@triton.jit
def _fused_sigmoid_bw(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-x, 20.0)))
    dsig = sig * (1.0 - sig)
    dx = dy * dsig
    tl.store(DX + offs, dx, mask=mask)


class FusedSigmoidBw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dy):
        assert x.is_contiguous() and dy.is_contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_sigmoid_bw[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx

    @staticmethod
    def backward(ctx, ddx):
        raise NotImplementedError("Second derivative not implemented")


def kernel_fn(x, dy):
    return FusedSigmoidBw.apply(x, dy)


def can_use_kernel(x, dy):
    return (x.is_cuda and dy.is_cuda and
            x.is_contiguous() and dy.is_contiguous() and
            x.shape == dy.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
