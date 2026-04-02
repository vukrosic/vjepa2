"""Fused Mish + HardSwish fusion kernel.

Pattern: y = mish(hardswish(x))
Fuses: hardswish + mish into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return torch.nn.functional.mish(torch.nn.functional.hardswish(x))


# --- KERNEL ---
@triton.jit
def _fused_mish_hardswish_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    hs = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    x1 = x * hs
    sp = tl.minimum(x1, 20.0)
    y = x1 * tl.tanh(sp)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_mish_hardswish_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    hs = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    x1 = x * hs
    sp = tl.minimum(x1, 20.0)
    tanh_sp = tl.tanh(sp)
    sig = 1.0 / (1.0 + tl.exp(-x1))
    dsp = sig * (1.0 - sig)
    # HardSwish derivative
    dhs = tl.where((x + 3.0) < 0.0, 0.0, tl.where((x + 3.0) > 6.0, 1.0, (2.0 * x + 6.0) / 6.0))
    # d(mish_hs)/dx = d_mish/d(hs) * d(hs)/dx = (tanh(sp) + x1 * dsp * dsp_dx1) * dhs
    # dsp_dx1 = sig * (1 - sig)
    # d(hs)/dx = (2x+6)/6
    dhs_dx = (2.0 * x + 6.0) / 6.0
    dhs_dx = tl.where((x + 3.0) < 0.0, 0.0, tl.where((x + 3.0) > 6.0, 0.0, dhs_dx))
    dx = dy * (tanh_sp + x1 * dsp * dhs_dx)
    tl.store(DX + offs, dx, mask=mask)


class FusedMishHardSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_mish_hardswish_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_mish_hardswish_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedMishHardSwish.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
