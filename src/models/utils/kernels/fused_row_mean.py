"""Fused Row Mean kernel.

Pattern: y = mean(x, dim=-1, keepdim=True)
Fuses: sum + divide into one reduction pass per row.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return x.mean(dim=-1, keepdim=True)


# --- KERNEL ---
@triton.jit
def _row_mean_fwd(X, Y, stride_row, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    X_ptr = X + row * stride_row
    Y_ptr = Y + row

    acc = tl.cast(0.0, tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        block_sum = tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc = acc + block_sum

    mean_val = acc / N
    tl.store(Y_ptr, mean_val)


class FusedRowMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        orig_shape = x.shape
        x2d = x.view(-1, x.shape[-1])
        rows, N = x2d.shape
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_N = min(BLOCK_N, 4096)
        y2d = torch.empty(rows, 1, dtype=x.dtype, device=x.device)
        _row_mean_fwd[(rows,)](
            x2d, y2d, x2d.stride(0), N=N, BLOCK_N=BLOCK_N,
            num_warps=min(16, max(1, BLOCK_N // 32))
        )
        return y2d.view(*orig_shape[:-1], 1)

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("RowMean backward not yet implemented")


def kernel_fn(x):
    return FusedRowMean.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
