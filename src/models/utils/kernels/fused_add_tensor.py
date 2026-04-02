"""Fused Add (two tensors) kernel.

Pattern: y = x + y_tensor
Fuses: addition into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, y_tensor):
    return x + y_tensor


# --- KERNEL ---
@triton.jit
def _fused_add_tensor_fwd(X, Y_T, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.load(Y_T + offs, mask=mask).to(tl.float32)
    out = x + y
    tl.store(Y + offs, out, mask=mask)


@triton.jit
def _fused_add_tensor_bwd(X, Y_T, DY, DX1, DX2, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    tl.store(DX1 + offs, dy, mask=mask)
    tl.store(DX2 + offs, dy, mask=mask)


class FusedAddTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y_tensor):
        assert x.is_contiguous() and y_tensor.is_contiguous()
        ctx.save_for_backward(y_tensor)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_add_tensor_fwd[grid](x, y_tensor, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y_tensor,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx1 = torch.empty_like(dy)
        dx2 = torch.empty_like(dy)
        N = dy.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_add_tensor_bwd[grid](dy, y_tensor, dy, dx1, dx2, N, BLOCK=BLOCK, num_warps=4)
        return dx1, dx2


def kernel_fn(x, y_tensor):
    return FusedAddTensor.apply(x, y_tensor)


def can_use_kernel(x, y_tensor):
    return (x.is_cuda and y_tensor.is_cuda and
            x.is_contiguous() and y_tensor.is_contiguous() and
            x.shape == y_tensor.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "y": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120), "y": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536), "y": (8, 256, 1536)},
}
