"""Fused Row RMS kernel.

Pattern: y = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
Fuses: square + mean + sqrt into one reduction pass per row.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, eps=1e-6):
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


# --- KERNEL ---
@triton.jit
def _row_rms_fwd(X, Y, stride_row, N: tl.constexpr, EPS: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    X_ptr = X + row * stride_row
    Y_ptr = Y + row

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc = tl.where(mask, x * x, 0.0)
        acc = tl.sum(acc, axis=0)

    rms = tl.sqrt(acc / N + EPS)
    tl.store(Y_ptr, rms)


class FusedRowRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps=1e-6):
        assert x.is_contiguous()
        orig_shape = x.shape
        x2d = x.view(-1, x.shape[-1])
        rows, N = x2d.shape
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_N = min(BLOCK_N, 4096)
        y2d = torch.empty(rows, 1, dtype=x.dtype, device=x.device)
        _row_rms_fwd[(rows,)](
            x2d, y2d, x2d.stride(0), N=N, EPS=eps, BLOCK_N=BLOCK_N,
            num_warps=min(16, max(1, BLOCK_N // 32))
        )
        return y2d.view(*orig_shape[:-1], 1)

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("RowRMS backward not yet implemented")


def kernel_fn(x, eps=1e-6):
    return FusedRowRMS.apply(x, eps)


def can_use_kernel(x, eps=1e-6):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "eps": 1e-6},
    "vit_h":  {"x": (2, 2048, 5120), "eps": 1e-6},
    "small":  {"x": (8, 256, 1536), "eps": 1e-6},
}
