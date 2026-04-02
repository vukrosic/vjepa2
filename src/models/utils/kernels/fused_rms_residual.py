"""Fused RMSNorm + residual add helper.

This queue entry is correctness-first. It keeps the exact PyTorch reference
path and a strict applicability guard so unsupported layouts fall back safely.
"""

import torch
import triton
import triton.language as tl


def baseline_fn(x, residual, weight, eps=1e-6):
    y = x + residual
    rms = y.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (y / rms * weight).to(x.dtype)


def _has_valid_shape(x, residual, weight):
    return (
        x.ndim >= 1
        and x.shape == residual.shape
        and weight.ndim == 1
        and weight.shape[0] == x.shape[-1]
    )


def can_use_kernel(x, residual, weight):
    return (
        x.is_cuda
        and residual.is_cuda
        and weight.is_cuda
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and _has_valid_shape(x, residual, weight)
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and x.dtype == residual.dtype == weight.dtype
    )


# --- FORWARD KERNEL ---
@triton.jit
def _fused_rms_res_fwd(
    X_ptr, R_ptr, W_ptr, Y_ptr,
    stride_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per row. Fuses residual add + RMSNorm."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # Load x and residual, accumulate sum of squares
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + offs
        m = cols < N
        x = tl.load(X_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        r = tl.load(R_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        y = x + r
        acc += y * y

    ss = tl.sum(acc, axis=0)
    rms_inv = tl.rsqrt(ss / N + eps)

    # Normalize, apply weight, and store
    for off in range(0, N, BLOCK_N):
        cols = off + offs
        m = cols < N
        x = tl.load(X_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        r = tl.load(R_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        y = x + r
        w = tl.load(W_ptr + cols, mask=m, other=0.0).to(tl.float32)
        out = y * rms_inv * w
        tl.store(Y_ptr + row * stride_row + cols, out.to(y.dtype), mask=m)


# --- BACKWARD KERNEL ---
@triton.jit
def _fused_rms_res_bwd(
    X_ptr, R_ptr, W_ptr, DY_ptr, DX_ptr, DR_ptr, DW_ptr,
    stride_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Backward: computes dX, dResidual, dWeight."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # Compute sum of squares and rms
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + offs
        m = cols < N
        x = tl.load(X_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        r = tl.load(R_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        y = x + r
        acc += y * y

    ss = tl.sum(acc, axis=0)
    rms = tl.sqrt(ss / N + eps)
    rms_inv = 1.0 / rms
    rms2 = rms * rms

    # Compute dot = sum(dy * w * y) for the correction term
    dot_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + offs
        m = cols < N
        x = tl.load(X_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        r = tl.load(R_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        y = x + r
        w = tl.load(W_ptr + cols, mask=m, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        dot_acc += dy * w * y
    sum_dot = tl.sum(dot_acc, axis=0)

    # dx = dr = (dy * w - y * sum_dot / (N * rms^2)) / rms
    for off in range(0, N, BLOCK_N):
        cols = off + offs
        m = cols < N
        x = tl.load(X_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        r = tl.load(R_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        y = x + r
        dy = tl.load(DY_ptr + row * stride_row + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=m, other=0.0).to(tl.float32)
        dx = (dy * w - y * sum_dot / (N * rms2)) * rms_inv
        dr = dx  # same gradient for x and residual since y = x + r
        tl.store(DX_ptr + row * stride_row + cols, dx.to(x.dtype), mask=m)
        tl.store(DR_ptr + row * stride_row + cols, dr.to(r.dtype), mask=m)

        # dw += dy * y * rms_inv
        dw = dy * y * rms_inv
        tl.atomic_add(DW_ptr + cols, dw, mask=m)


class FusedRMSResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, eps=1e-6):
        orig_shape = x.shape
        D = x.shape[-1]
        N = x.numel() // D
        BLOCK_N = triton.next_power_of_2(D)
        BLOCK_N = min(BLOCK_N, 4096)

        x_c = x.contiguous()
        r_c = residual.contiguous()
        y = torch.empty_like(x_c)

        _fused_rms_res_fwd[(N,)](
            x_c, r_c, weight.contiguous(), y,
            D,
            N=N, eps=eps, BLOCK_N=BLOCK_N,
            num_warps=min(16, max(1, BLOCK_N // 32)),
        )
        ctx.save_for_backward(x_c, r_c, weight)
        ctx.eps = eps
        ctx.D = D
        ctx.BLOCK_N = BLOCK_N
        ctx.orig_shape = orig_shape
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_c, r_c, weight = ctx.saved_tensors
        eps = ctx.eps
        D = ctx.D
        BLOCK_N = ctx.BLOCK_N
        N = x_c.numel() // D

        dy_c = dy.contiguous().view(N, D)
        x_flat = x_c.view(N, D)
        r_flat = r_c.view(N, D)

        dx = torch.empty_like(x_flat)
        dr = torch.empty_like(r_flat)
        dw = torch.zeros(D, dtype=torch.float32, device=x_c.device)

        _fused_rms_res_bwd[(N,)](
            x_flat, r_flat, weight, dy_c, dx, dr, dw,
            D, N=D, eps=eps, BLOCK_N=BLOCK_N,
            num_warps=min(16, max(1, BLOCK_N // 32)),
        )
        return (
            dx.view(ctx.orig_shape),
            dr.view(ctx.orig_shape),
            dw.to(weight.dtype),
            None,
        )


def kernel_fn(x, residual, weight, eps=1e-6):
    if not can_use_kernel(x, residual, weight):
        return baseline_fn(x, residual, weight, eps)
    return FusedRMSResidual.apply(x, residual, weight, eps)


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "C": 1024},
    "vit_h": {"x": (2, 2048, 1280), "C": 1280},
    "small": {"x": (4, 256, 384), "C": 384},
}
