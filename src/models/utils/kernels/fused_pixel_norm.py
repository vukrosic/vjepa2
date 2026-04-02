"""Fused Pixel Normalization kernel.

Pattern: y = x / sqrt(mean(x^2) + eps)
Normalizes across spatial (H, W) dims per channel.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, eps=1e-6):
    # Normalize over last 3 dims (C is first of those 3)
    dims = tuple(range(x.dim() - 3, x.dim()))
    mean_sq = x.pow(2).mean(dim=dims, keepdim=True)
    return x / (mean_sq + eps).sqrt()


# --- KERNEL ---
@triton.jit
def _pixel_norm_fwd(X, Y, stride_c, H: tl.constexpr, W: tl.constexpr, EPS: tl.constexpr, BLOCK_H: tl.constexpr):
    c = tl.program_id(0)
    X_c = X + c * stride_c
    Y_c = Y + c * stride_c

    # Compute mean of squares
    acc = tl.zeros([BLOCK_H, W], dtype=tl.float32)
    for row in range(0, H, BLOCK_H):
        rows = row + tl.arange(0, BLOCK_H)
        mask = rows < H
        for col in range(W):
            x = tl.load(X_c + rows * W + col, mask=mask, other=0.0).to(tl.float32)
            acc = tl.where(mask, acc + x * x, acc)
    ss = tl.sum(acc, axis=0)
    mean_sq = ss / (H * W)
    norm = tl.rsqrt(mean_sq + EPS)

    # Normalize
    for row in range(0, H, BLOCK_H):
        rows = row + tl.arange(0, BLOCK_H)
        mask = rows < H
        for col in range(W):
            x = tl.load(X_c + rows * W + col, mask=mask, other=0.0).to(tl.float32)
            tl.store(Y_c + rows * W + col, x * norm, mask=mask)


class FusedPixelNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps=1e-6):
        assert x.is_contiguous()
        assert x.dim() >= 3
        C = x.shape[-3]
        H = x.shape[-2]
        W = x.shape[-1]
        BLOCK_H = triton.next_power_of_2(H)
        BLOCK_H = min(BLOCK_H, 128)
        y = torch.empty_like(x)
        _pixel_norm_fwd[(C,)](x, y, x.stride(-3), H, W, eps, BLOCK_H, num_warps=8)
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("PixelNorm backward not yet implemented")


def kernel_fn(x, eps=1e-6):
    return FusedPixelNorm.apply(x, eps)


def can_use_kernel(x, eps=1e-6):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dim() >= 3 and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 16, 16)},
    "vit_h":  {"x": (2, 2048, 14, 14)},
    "small":  {"x": (8, 384, 8, 8)},
}
