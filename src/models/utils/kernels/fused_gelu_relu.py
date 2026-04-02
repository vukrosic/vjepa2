"""Fused gelu_relu kernel.

Pattern: torch.nn.functional.gelu(x) + torch.nn.functional.relu(x)
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.gelu(x) + torch.nn.functional.relu(x)


@triton.jit
def _fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    sqrt_2_over_pi = 0.7978845608028654
    k = 0.044542
    g = 0.5 * v * (1.0 + tl.tanh(sqrt_2_over_pi * v * (1.0 + k * v * v)))
    r = tl.where(v > 0, v, 0.0)
    y = g + r
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    sqrt_2_over_pi = 0.7978845608028654
    k = 0.044542
    tanh_arg = sqrt_2_over_pi * v * (1.0 + k * v * v)
    tanh_g = tl.tanh(tanh_arg)
    dtanh_g = 1.0 - tanh_g * tanh_g
    dg_dv = 0.5 * (1.0 + tanh_g) + 0.5 * v * dtanh_g * sqrt_2_over_pi * (1.0 + 3.0 * k * v * v)
    dr_dv = tl.where(v > 0, 1.0, 0.0)
    dx = dy * (dg_dv + dr_dv)
    tl.store(DX + offs, dx, mask=mask)


class GeluRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(g, v)
        return y

    @staticmethod
    def backward(ctx, dy):
        g, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(g)
        N = g.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _bwd[grid](g, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return GeluRelu.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
