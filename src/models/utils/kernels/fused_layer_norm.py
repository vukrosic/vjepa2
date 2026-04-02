"""Fused LayerNorm kernel.

Pattern: y = (x - mean) / sqrt(var + eps) * weight + bias
RMSNorm variant: y = x / sqrt(var + eps) * weight
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, weight, bias, eps=1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)


@triton.jit
def _layer_norm_fwd(X, Y, W, B, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x) / N
    var = tl.sum((x - mean) * (x - mean)) / N
    y = (x - mean) / tl.sqrt(var + EPS) * w + b
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _layer_norm_bwd(X, Y, DY, DX, DW, DB, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
    mean = tl.sum(x) / N
    var = tl.sum((x - mean) * (x - mean)) / N
    inv_std = 1.0 / tl.sqrt(var + EPS)
    dx = w * inv_std * dy
    tl.store(DX + offs, dx, mask=mask)
    tl.store(DW + offs, (x - mean) * inv_std * dy, mask=mask)
    tl.store(DB + offs, dy, mask=mask)


class FusedLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _layer_norm_fwd[(n_blocks,)](x, y, weight, bias, N, eps, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x, weight, bias)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias = ctx.saved_tensors
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        db = torch.empty_like(bias)
        N = x.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _layer_norm_bwd[(n_blocks,)](x, dy, dx, dw, db, N, ctx.eps, BLOCK=BLOCK, num_warps=4)
        return dx, dw, db, None


def kernel_fn(x, weight, bias, eps=1e-5):
    return FusedLayerNorm.apply(x, weight, bias, eps)


def can_use_kernel(x, weight, bias):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "weight": (4096,), "bias": (4096,)},
    "vit_h":  {"x": (2, 2048, 5120), "weight": (5120,), "bias": (5120,)},
    "small":  {"x": (8, 256, 1536), "weight": (1536,), "bias": (1536,)},
}
