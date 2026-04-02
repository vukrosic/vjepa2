"""Fused Normalize kernel.

Pattern: y = x / (||x||_p + eps)
L2-normalizes a tensor along a given dim.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, dim=-1, eps=1e-5):
    return torch.nn.functional.normalize(x, p=2.0, dim=dim, eps=eps)


# --- KERNEL ---
@triton.jit
def _normalize_fwd(X, Y, stride_row, N: tl.constexpr, EPS: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    X_ptr = X + row * stride_row
    Y_ptr = Y + row * stride_row

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc = tl.where(mask, acc + x * x, 0.0)
        acc = tl.sum(acc, axis=0)
    norm = tl.sqrt(acc + EPS)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x / norm
        tl.store(Y_ptr + cols, y, mask=mask)


class FusedNormalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=-1, eps=1e-5):
        assert x.is_contiguous()
        orig_shape = x.shape
        x2d = x.view(-1, x.shape[dim])
        rows, N = x2d.shape
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_N = min(BLOCK_N, 4096)
        y2d = torch.empty_like(x2d)
        _normalize_fwd[(rows,)](x2d, y2d, x2d.stride(0), N, eps, BLOCK_N, num_warps=min(16, max(1, BLOCK_N // 32)))
        return y2d.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("Normalize backward not yet implemented")


def kernel_fn(x, dim=-1, eps=1e-5):
    return FusedNormalize.apply(x, dim, eps)


def can_use_kernel(x, dim=-1, eps=1e-5):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
