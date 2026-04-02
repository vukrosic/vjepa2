"""Fused RMSNorm kernel.

RMSNorm = x / rms(x) * weight, where rms = sqrt(mean(x^2) + eps).
Fuses the rms computation and normalization into a single pass per row.
Used in some V-JEPA variants instead of LayerNorm.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, weight, eps=1e-6):
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return (x / rms * weight).to(x.dtype)


# --- FORWARD KERNEL ---
@triton.jit
def _fused_rms_norm_fwd(
    X, W, Y,
    stride_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr = X + row * stride_row
    Y_ptr = Y + row * stride_row

    # Compute sum of squares
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc += x * x

    ss = tl.sum(acc, axis=0)
    rms_inv = tl.rsqrt(ss / N + eps)

    # Normalize and apply weight
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rms_inv * w
        tl.store(Y_ptr + cols, y, mask=mask)


# --- BACKWARD KERNEL ---
@triton.jit
def _fused_rms_norm_bwd(
    X, W, DY, DX, DW,
    stride_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr = X + row * stride_row
    DY_ptr = DY + row * stride_row
    DX_ptr = DX + row * stride_row

    # Compute rms for this row
    ss = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        ss += x * x
    sum_sq = tl.sum(ss, axis=0)
    rms = tl.sqrt(sum_sq / N + eps)
    rms_inv = 1.0 / rms

    # Compute sum(dy * w * x) for the correction term
    dot = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dot += dy * w * x
    sum_dot = tl.sum(dot, axis=0)

    # dx = (dy * w - x * sum_dot / (N * rms^2)) / rms
    # dx = dy * w * rms_inv - x * sum_dot * rms_inv / (N * rms^2)
    # Simplify: rms_inv^2 = 1/rms^2
    # dx = rms_inv * (dy * w - x * sum_dot / (N * rms^2))
    # Note: rms^2 = sum_sq/N + eps, and rms_inv = 1/rms
    rms2 = rms * rms

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dx = (dy * w - x * sum_dot / (N * rms2)) * rms_inv
        tl.store(DX_ptr + cols, dx, mask=mask)

        # Accumulate dw: dw[j] += dy[row,j] * x[row,j] * rms_inv
        # We do atomic add to handle multiple rows
        dw = dy * x * rms_inv
        tl.atomic_add(DW + cols, dw, mask=mask)


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        assert x.is_contiguous()
        assert weight.is_contiguous()
        orig_shape = x.shape
        x2d = x.view(-1, x.shape[-1])
        rows, N = x2d.shape
        BLOCK_N = triton.next_power_of_2(N)
        # Clamp block size to 4096 to avoid shared memory issues
        BLOCK_N = min(BLOCK_N, 4096)
        y2d = torch.empty_like(x2d)
        _fused_rms_norm_fwd[(rows,)](
            x2d, weight, y2d,
            x2d.stride(0),
            N=N, eps=eps, BLOCK_N=BLOCK_N,
            num_warps=min(16, max(1, BLOCK_N // 32)),
        )
        ctx.save_for_backward(x2d, weight)
        ctx.eps = eps
        ctx.N = N
        ctx.BLOCK_N = BLOCK_N
        return y2d.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, weight = ctx.saved_tensors
        dy2d = dy.contiguous().view_as(x2d)
        rows, N = x2d.shape
        BLOCK_N = ctx.BLOCK_N
        dx2d = torch.empty_like(x2d)
        dw = torch.zeros(N, dtype=torch.float32, device=x2d.device)
        _fused_rms_norm_bwd[(rows,)](
            x2d, weight, dy2d, dx2d, dw,
            x2d.stride(0),
            N=N, eps=ctx.eps, BLOCK_N=BLOCK_N,
            num_warps=min(16, max(1, BLOCK_N // 32)),
        )
        return dx2d.view_as(dy), dw.to(weight.dtype), None


def kernel_fn(x, weight, eps=1e-6):
    return FusedRMSNorm.apply(x, weight, eps)


def can_use_kernel(x, weight):
    return (
        x.is_cuda and weight.is_cuda
        and x.is_contiguous() and weight.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.shape[-1] == weight.shape[0]
    )


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "D": 1024},
    "vit_h": {"x": (2, 2048, 1280), "D": 1280},
    "small": {"x": (8, 256, 384), "D": 384},
}
