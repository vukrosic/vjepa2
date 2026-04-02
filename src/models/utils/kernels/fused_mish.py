"""Fused Mish activation kernel.

Pattern: x * tanh(softplus(x)) = x * tanh(log(exp(x) + 1))
Fuses: softplus + tanh + multiply into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x):
    return torch.nn.functional.mish(x)


# --- KERNEL ---
@triton.jit
def _fused_mish_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    # softplus = log(exp(x) + 1); use tl.abs + tl.log(1 + tl.exp(-tl.abs(x))) for stability
    # mish: x * tanh(softplus(x))
    sp = tl.log(tl.exp(tl.min(x, 20.0)) + 1.0)  # clamp for stability
    t = tl.tanh(sp)
    y = x * t
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_mish_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    sp = tl.log(tl.exp(tl.min(x, 20.0)) + 1.0)
    t = tl.tanh(sp)
    # d(tanh(sp))/dsp = 1 - tanh(sp)^2 = 1 - t^2
    # d(softplus)/dx = sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(tl.min(-x, 20.0)))
    dtanh_dsp = 1.0 - t * t
    dsp_dx = sig
    # d(out)/dx = tanh(sp) + x * dtanh_dsp * dsp_dx = t + x * (1 - t^2) * sig
    dx = dy * (t + x * dtanh_dsp * dsp_dx)
    tl.store(DX + offs, dx, mask=mask)


class FusedMish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_mish_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_mish_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedMish.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
