"""Fused GELU with tanh kernel.

Pattern: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
Exact GELU formula using tanh (not erf).
Fuses: polynomial + tanh + multiply into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


# --- KERNEL ---
@triton.jit
def _fused_gelu_tanh_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    k = 0.7978845608028654  # sqrt(2/pi)
    c = 0.044715
    tanh_arg = k * (x + c * x * x * x)
    tanh_val = tl.tanh(tanh_arg)
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_gelu_tanh_bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    k = 0.7978845608028654
    c = 0.044715
    tanh_arg = k * (x + c * x * x * x)
    tanh_val = tl.tanh(tanh_arg)
    # dGELU/dx = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * k * (1 + 3*c*x^2)
    # sech^2(z) = 1 - tanh^2(z)
    d_tanh = 1.0 - tanh_val * tanh_val
    dx = 0.5 * (1.0 + tanh_val) + 0.5 * x * d_tanh * k * (1.0 + 3.0 * c * x * x)
    dx = dy * dx
    tl.store(DX + offs, dx, mask=mask)


class FusedGELUTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_tanh_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_tanh_bwd[grid](x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedGELUTanh.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
