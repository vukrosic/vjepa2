"""Fused HardTanh kernel.

Pattern: hard_tanh(x) = clamp(x, -1, 1)
Fuses: clamp into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return torch.nn.functional.hardtanh(x)


# --- KERNEL ---
@triton.jit
def _fused_hard_tanh_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.where(x > 1.0, 1.0, x)
    y = tl.where(y < -1.0, -1.0, y)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_hard_tanh_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    dx = tl.where((x >= -1.0) & (x <= 1.0), dy, 0.0)
    tl.store(DX + offs, dx, mask=mask)


class FusedHardTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_hard_tanh_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_hard_tanh_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedHardTanh.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
