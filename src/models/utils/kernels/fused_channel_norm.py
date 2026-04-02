"""Fused Channel Normalization kernel.

Pattern: y = (x - mean) / sqrt(var + eps) * weight + bias
Normalizes over channel dimension (C).
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, weight, bias, eps=1e-5):
    dims = tuple(range(1, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt() * weight + bias


# --- KERNEL ---
@triton.jit
def _channel_norm_fwd(X, Y, W, B, stride_b, C: tl.constexpr, HWN: tl.constexpr, EPS: tl.constexpr, BLOCK_C: tl.constexpr):
    b = tl.program_id(0)
    X_b = X + b * stride_b
    Y_b = Y + b * stride_b

    # Compute mean per channel
    for ch in range(C):
        acc = 0.0
        for off in range(0, HWN, BLOCK_C):
            offs = off + tl.arange(0, BLOCK_C)
            mask = offs < HWN
            x = tl.load(X_b + ch * HWN + offs, mask=mask, other=0.0).to(tl.float32)
            acc = tl.sum(x) if off == 0 else acc + tl.sum(x)
        mean = acc / HWN

        acc = 0.0
        for off in range(0, HWN, BLOCK_C):
            offs = off + tl.arange(0, BLOCK_C)
            mask = offs < HWN
            x = tl.load(X_b + ch * HWN + offs, mask=mask, other=0.0).to(tl.float32)
            d = x - mean
            acc = tl.sum(d * d) if off == 0 else acc + tl.sum(d * d)
        var = acc / HWN
        inv_std = 1.0 / tl.sqrt(var + EPS)
        w = tl.load(W + ch).to(tl.float32)
        b = tl.load(B + ch).to(tl.float32)

        for off in range(0, HWN, BLOCK_C):
            offs = off + tl.arange(0, BLOCK_C)
            mask = offs < HWN
            x = tl.load(X_b + ch * HWN + offs, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * inv_std * w + b
            tl.store(Y_b + ch * HWN + offs, y, mask=mask)


class FusedChannelNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        assert x.is_contiguous()
        B, C = x.shape[:2]
        HWN = x.numel() // (B * C)
        BLOCK_C = triton.next_power_of_2(HWN)
        BLOCK_C = min(BLOCK_C, 4096)
        y = torch.empty_like(x)
        _channel_norm_fwd[(B,)](
            x, y, weight, bias, x.stride(0), C, HWN, eps, BLOCK_C, num_warps=8
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("ChannelNorm backward not yet implemented")


def kernel_fn(x, weight, bias, eps=1e-5):
    return FusedChannelNorm.apply(x, weight, bias, eps)


def can_use_kernel(x, weight, bias, eps=1e-5):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 256)},
    "vit_h":  {"x": (2, 2048, 196)},
    "small":  {"x": (8, 384, 64)},
}
