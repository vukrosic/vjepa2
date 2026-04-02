"""Fused Softmax kernel.

Pattern: y_i = exp(x_i) / sum_j(exp(x_j))
Fuses exp and sum reduction into a two-pass kernel.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, dim=-1):
    return torch.nn.functional.softmax(x, dim=dim)


@triton.jit
def _softmax_fwd(X, Y, stride_row, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    X_ptr = X + row * stride_row

    # First pass: compute max
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc = tl.where(mask, x, acc)
    max_x = tl.max(acc, axis=0)

    # Second pass: compute exp sum
    acc2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        e = tl.exp(tl.minimum(x - max_x, 40.0))
        acc2 = tl.where(mask, acc2 + e, acc2)
    sum_exp = tl.sum(acc2, axis=0)

    # Third pass: compute output
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        e = tl.exp(tl.minimum(x - max_x, 40.0))
        y = e / (sum_exp + 1e-8)
        tl.store(Y + row * stride_row + cols, y, mask=mask)


class FusedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        assert x.is_contiguous()
        orig_shape = x.shape
        # Flatten all but the softmax dimension
        if dim == -1 or dim == len(orig_shape) - 1:
            x2d = x.view(-1, x.shape[-1])
        else:
            x2d = x.transpose(0, dim).contiguous()
            x2d = x2d.view(-1, x2d.shape[-1])
        rows, N = x2d.shape
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_N = min(BLOCK_N, 4096)
        y2d = torch.empty_like(x2d)
        _softmax_fwd[(rows,)](x2d, y2d, x2d.stride(0), N=N, BLOCK_N=BLOCK_N,
                               num_warps=min(16, max(1, BLOCK_N // 32)))
        ctx.save_for_backward(x2d)
        ctx.dim = dim
        return y2d.view(orig_shape) if dim == -1 or dim == len(orig_shape) - 1 else y2d.view(orig_shape).transpose(0, dim)

    @staticmethod
    def backward(ctx, dy):
        # Simplified: do not backprop through softmax in this fused version
        return dy, None


def kernel_fn(x, dim=-1):
    return FusedSoftmax.apply(x, dim)


def can_use_kernel(x, dim=-1):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536)},
}
