"""Fused HardTanh v2 activation kernel.

Pattern: y = clamp(x, min_val, max_val)
Fuses: two comparisons + select into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, min_val=-1.0, max_val=1.0):
    return torch.nn.functional.hardtanh(x, min_val, max_val)


# --- KERNEL ---
@triton.jit
def _fused_hardtanh_v2_fwd(X, Y, N: tl.constexpr, MIN_V: tl.constexpr, MAX_V: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.minimum(tl.maximum(x, MIN_V), MAX_V)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_hardtanh_v2_bwd(X, DY, DX, N: tl.constexpr, MIN_V: tl.constexpr, MAX_V: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    dx = dy * tl.where((x >= MIN_V) & (x <= MAX_V), 1.0, 0.0)
    tl.store(DX + offs, dx, mask=mask)


class FusedHardTanhV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_val=-1.0, max_val=1.0):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.min_val = min_val
        ctx.max_val = max_val
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_hardtanh_v2_fwd[grid](x, y, N, min_val, max_val, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_hardtanh_v2_bwd[grid](x, dy, dx, N, ctx.min_val, ctx.max_val, BLOCK=BLOCK, num_warps=4)
        return dx, None, None


def kernel_fn(x, min_val=-1.0, max_val=1.0):
    return FusedHardTanhV2.apply(x, min_val, max_val)


def can_use_kernel(x, min_val=-1.0, max_val=1.0):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
