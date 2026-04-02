"""Fused mish_relu kernel.

Pattern: torch.nn.functional.mish(torch.nn.functional.relu(x))
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.mish(torch.nn.functional.relu(x))


@triton.jit
def _fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.where(v > 0, v, 0.0)
    sp = tl.log(1.0 + tl.exp(tl.minimum(r, 20.0)))
    e2sp = tl.exp(tl.minimum(2.0 * sp, 40.0))
    tanh_sp = (e2sp - 1.0) / (e2sp + 1.0)
    y = r * tanh_sp
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.where(v > 0, v, 0.0)
    dr_dv = tl.where(v > 0, 1.0, 0.0)
    e2sp = tl.exp(tl.minimum(2.0 * sp, 40.0))
    tanh_sp = (e2sp - 1.0) / (e2sp + 1.0)
    sig_sp = 1.0 / (1.0 + tl.exp(tl.minimum(-r, 20.0)))
    dtanh_sp = 1.0 - tanh_sp * tanh_sp
    dsp_dr = sig_sp
    d_out_dv = dr_dv * (tanh_sp + r * dtanh_sp * dsp_dr)
    dx = dy * d_out_dv
    tl.store(DX + offs, dx, mask=mask)


class MishRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(r, sp, v)
        return y

    @staticmethod
    def backward(ctx, dy):
        r, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(r)
        N = r.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _bwd[grid](r, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return MishRelu.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
