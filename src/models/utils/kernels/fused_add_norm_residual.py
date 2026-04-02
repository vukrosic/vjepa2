"""Fused add + LayerNorm for two-branch residual blocks.

Pattern: out = LayerNorm(x + branch1 + branch2)
Fuses: 3-input add + LayerNorm in one pass per row.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x, b1, b2, weight, bias, eps=1e-5):
    y = x + b1 + b2
    return torch.nn.functional.layer_norm(y, (y.shape[-1],), weight, bias, eps)


# --- KERNEL ---
@triton.jit
def _fused_add_norm_fwd(X, B1, B2, W, B, Y, N: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    # Load and sum x + b1 + b2
    sum_val = 0.0
    for d in range(D):
        idx = pid * D + d
        x0 = tl.load(X + idx, mask=mask_d, other=0.0).to(tl.float32)
        x1 = tl.load(B1 + idx, mask=mask_d, other=0.0).to(tl.float32)
        x2 = tl.load(B2 + idx, mask=mask_d, other=0.0).to(tl.float32)
        sum_val += x0 + x1 + x2
        mask_d = (offs_d + d + 1) < D
    mean = sum_val / D

    # Compute variance
    sq_sum = 0.0
    mask_d = offs_d < D
    for d in range(D):
        idx = pid * D + d
        x0 = tl.load(X + idx, mask=mask_d, other=0.0).to(tl.float32)
        x1 = tl.load(B1 + idx, mask=mask_d, other=0.0).to(tl.float32)
        x2 = tl.load(B2 + idx, mask=mask_d, other=0.0).to(tl.float32)
        val = x0 + x1 + x2 - mean
        sq_sum += val * val
        mask_d = (offs_d + d + 1) < D
    var = sq_sum / D
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Normalize and apply weight/bias
    mask_d = offs_d < D
    for d in range(D):
        idx = pid * D + d
        x0 = tl.load(X + idx, mask=mask_d, other=0.0).to(tl.float32)
        x1 = tl.load(B1 + idx, mask=mask_d, other=0.0).to(tl.float32)
        x2 = tl.load(B2 + idx, mask=mask_d, other=0.0).to(tl.float32)
        val = (x0 + x1 + x2 - mean) * inv_std
        w = tl.load(W + d, mask=mask_d, other=0.0).to(tl.float32)
        b = tl.load(B + d, mask=mask_d, other=0.0).to(tl.float32)
        out = val * w + b
        tl.store(Y + idx, out, mask=mask_d)
        mask_d = (offs_d + d + 1) < D


@triton.jit
def _fused_add_norm_bwd(X, B1, B2, W, DY, DX, DB1, DB2, DW, DB, N: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Recompute mean and variance
    sum_val = 0.0
    for d in range(D):
        idx = pid * D + d
        x0 = tl.load(X + idx, mask=mask_d, other=0.0).to(tl.float32)
        x1 = tl.load(B1 + idx, mask=mask_d, other=0.0).to(tl.float32)
        x2 = tl.load(B2 + idx, mask=mask_d, other=0.0).to(tl.float32)
        sum_val += x0 + x1 + x2
        mask_d = (offs_d + d + 1) < D
    mean = sum_val / D

    sq_sum = 0.0
    mask_d = offs_d < D
    for d in range(D):
        idx = pid * D + d
        x0 = tl.load(X + idx, mask=mask_d, other=0.0).to(tl.float32)
        x1 = tl.load(B1 + idx, mask=mask_d, other=0.0).to(tl.float32)
        x2 = tl.load(B2 + idx, mask=mask_d, other=0.0).to(tl.float32)
        val = x0 + x1 + x2 - mean
        sq_sum += val * val
        mask_d = (offs_d + d + 1) < D
    var = sq_sum / D
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Backward: dL/dx, dL/db1, dL/db2, dL/dw, dL/db
    for d in range(D):
        idx = pid * D + d
        dy = tl.load(DY + idx, mask=mask_d, other=0.0).to(tl.float32)
        w = tl.load(W + d, mask=mask_d, other=0.0).to(tl.float32)
        # dval/dx = w * inv_std
        d_out = dy * w * inv_std
        tl.store(DX + idx, d_out, mask=mask_d)
        tl.store(DB1 + idx, d_out, mask=mask_d)
        tl.store(DB2 + idx, d_out, mask=mask_d)
        # dL/dw += dy * val
        x0 = tl.load(X + idx, mask=mask_d, other=0.0).to(tl.float32)
        x1 = tl.load(B1 + idx, mask=mask_d, other=0.0).to(tl.float32)
        x2 = tl.load(B2 + idx, mask=mask_d, other=0.0).to(tl.float32)
        val = (x0 + x1 + x2 - mean) * inv_std
        dw = dy * val
        tl.atomic_add(DW + d, dw, mask=mask_d)
        tl.atomic_add(DB + d, dy, mask=mask_d)
        mask_d = (offs_d + d + 1) < D


class FusedAddNormResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b1, b2, weight, bias, eps=1e-5):
        assert x.is_contiguous() and b1.is_contiguous() and b2.is_contiguous()
        B, D = x.shape
        ctx.save_for_backward(x, b1, b2, weight, bias)
        ctx.eps = eps
        y = torch.empty_like(x)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B,)
        _fused_add_norm_fwd[grid](x, b1, b2, weight, bias, y, B, D, eps, BLOCK_D=BLOCK_D, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, b1, b2, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        B, D = x.shape
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        db1 = torch.empty_like(b1)
        db2 = torch.empty_like(b2)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(bias)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B,)
        _fused_add_norm_bwd[grid](x, b1, b2, weight, dy, dx, db1, db2, dw, db, B, D, eps, BLOCK_D=BLOCK_D, num_warps=4)
        return dx, db1, db2, dw, db, None


def kernel_fn(x, b1, b2, weight, bias, eps=1e-5):
    return FusedAddNormResidual.apply(x, b1, b2, weight, bias, eps)


def can_use_kernel(x, b1, b2, weight, bias):
    return (x.is_cuda and b1.is_cuda and b2.is_cuda and weight.is_cuda and bias.is_cuda and
            x.is_contiguous() and b1.is_contiguous() and b2.is_contiguous() and
            x.shape == b1.shape == b2.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 1024), "b1": (2, 1024, 1024), "b2": (2, 1024, 1024), "D": 1024},
    "vit_h":  {"x": (2, 2048, 1280), "b1": (2, 2048, 1280), "b2": (2, 2048, 1280), "D": 1280},
    "small":  {"x": (8, 256, 384),   "b1": (8, 256, 384),   "b2": (8, 256, 384),   "D": 384},
}
