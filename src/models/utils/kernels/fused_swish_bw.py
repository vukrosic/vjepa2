"""Fused Swish backward-only kernel (gradient).

Pattern: dx = dy * (swish(x) + x * sigmoid(x) * (1 - sigmoid(x)))
Fuses: swish gradient computation into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, dy):
    """Backward for swish: d/dx[x*sigmoid(x)] = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))"""
    sigmoid = torch.sigmoid(x)
    return dy * (sigmoid + x * sigmoid * (1 - sigmoid))


# --- KERNEL ---
@triton.jit
def _fused_swish_bw(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-x, 20.0)))
    swish_grad = sig + x * sig * (1.0 - sig)
    dx = dy * swish_grad
    tl.store(DX + offs, dx, mask=mask)


class FusedSwishBw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dy):
        assert x.is_contiguous() and dy.is_contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_swish_bw[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx

    @staticmethod
    def backward(ctx, ddx):
        raise NotImplementedError("Second derivative not implemented")


def kernel_fn(x, dy):
    return FusedSwishBw.apply(x, dy)


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
