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
    # softplus: log(exp(x) + 1), clamped for stability
    sp = tl.log(tl.exp(tl.minimum(x, 20.0)) + 1.0)
    # tanh(sp) = (exp(2*sp) - 1) / (exp(2*sp) + 1)
    e2sp = tl.exp(tl.minimum(2.0 * sp, 40.0))
    tanh_sp = (e2sp - 1.0) / (e2sp + 1.0)
    y = x * tanh_sp
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_mish_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    sp = tl.log(tl.exp(tl.minimum(x, 20.0)) + 1.0)
    e2sp = tl.exp(tl.minimum(2.0 * sp, 40.0))
    tanh_sp = (e2sp - 1.0) / (e2sp + 1.0)
    # sigmoid(x) for softplus gradient: 1/(1+exp(-x_clamped))
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-x, 20.0)))
    # dtanh/dsp = 1 - tanh^2
    dtanh_dsp = 1.0 - tanh_sp * tanh_sp
    dsp_dx = sig
    # d(out)/dx = tanh(sp) + x * dtanh_dsp * dsp_dx
    dx = dy * (tanh_sp + x * dtanh_dsp * dsp_dx)
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
