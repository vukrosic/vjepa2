"""Fused Group Normalization kernel.

Pattern: y = (x - mean) / sqrt(var + eps) * weight + bias
Groups channels into G groups and normalizes per group.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, weight, bias, G=32, eps=1e-5):
    return torch.nn.functional.group_norm(x, G, weight=weight, bias=bias, eps=eps)


# --- KERNEL ---
@triton.jit
def _group_norm_fwd(X, Y, W, B, stride_b, C: tl.constexpr, G: tl.constexpr, HWN: tl.constexpr, EPS: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    g = tl.program_id(1)
    X_ptr = X + b * stride_b + g * (C // G) * HWN
    Y_ptr = Y + b * stride_b + g * (C // G) * HWN

    # Compute mean
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    N = C // G * HWN
    for off in range(0, N, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = tl.where(mask, acc + x, 0.0)
        acc = tl.sum(acc, axis=0)
    mean = acc / N

    # Compute variance
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        d = x - mean
        acc = tl.where(mask, acc + d * d, 0.0)
        acc = tl.sum(acc, axis=0)
    var = acc / N
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Load affine params
    c_start = g * (C // G)
    w = tl.load(W + c_start:c_start + C // G)
    b = tl.load(B + c_start:c_start + C // G)

    # Normalize + affine
    for off in range(0, N, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        ci = offs // HWN
        y = (x - mean) * inv_std * w[ci] + b[ci]
        tl.store(Y_ptr + offs, y, mask=mask)


class FusedGroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, G=32, eps=1e-5):
        assert x.is_contiguous()
        B, C = x.shape[:2]
        HWN = x.numel() // (B * C)
        BLOCK_N = triton.next_power_of_2(HWN)
        BLOCK_N = min(BLOCK_N, 4096)
        y = torch.empty_like(x)
        _group_norm_fwd[(B, G)](
            x, y, weight, bias, x.stride(0), C, G, HWN, eps, BLOCK_N,
            num_warps=min(16, max(1, BLOCK_N // 32))
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("GroupNorm backward not yet implemented")


def kernel_fn(x, weight, bias, G=32, eps=1e-5):
    return FusedGroupNorm.apply(x, weight, bias, G, eps)


def can_use_kernel(x, weight, bias, G=32, eps=1e-5):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            x.shape[1] % G == 0)


SHAPES = {
    "vit_l":  {"x": (2, 1024, 256), "G": 32},
    "vit_h":  {"x": (2, 2048, 196), "G": 32},
    "small":  {"x": (8, 384, 64), "G": 32},
}
