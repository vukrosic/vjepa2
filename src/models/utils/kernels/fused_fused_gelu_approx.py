"""Fused GELU-approx kernel.
Pattern: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl

def baseline_fn(x):
    return torch.nn.functional.gelu(x)

@triton.jit
def _gelu_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    c0 = 0.7978845608028654
    c1 = 0.044715
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N: break
        v = tl.load(X + offs).to(tl.float32)
        inner = c0 * (v + c1 * v * v * v)
        tanh_i = tl.tanh(inner)
        tl.store(Y + offs, 0.5 * v * (1.0 + tanh_i))

@triton.jit
def _gelu_bwd(X, Y, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    c0 = 0.7978845608028654
    c1 = 0.044715
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N: break
        x = tl.load(X + offs).to(tl.float32)
        dy = tl.load(DY + offs).to(tl.float32)
        inner = c0 * (x + c1 * x * x * x)
        tanh_i = tl.tanh(inner)
        d_tanh = 1.0 - tanh_i * tanh_i
        dinner = c0 * (1.0 + 3.0 * c1 * x * x)
        tl.store(DX + offs, dy * (0.5 * (1.0 + tanh_i) + 0.5 * x * d_tanh * dinner))

class FusedGeluApprox(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel(); BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _gelu_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y
    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel(); BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _gelu_bwd[grid](x, x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx

def kernel_fn(x):
    return FusedGeluApprox.apply(x)

def can_use_kernel(x):
    return x.is_cuda and x.is_contiguous() and x.dtype in (torch.float16, torch.float32, torch.bfloat16)

SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
