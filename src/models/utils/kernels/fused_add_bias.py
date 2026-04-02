"""Fused Add-Bias kernel.

Pattern: y = x + bias  (broadcast bias across last dimension)
Fuses: broadcast add into elementwise pass.
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, bias):
    return x + bias


# --- KERNEL ---
@triton.jit
def _add_bias_fwd(X, Bias, Y, N: tl.constexpr, LAST_DIM: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        bias_idx = offs % LAST_DIM
        v = tl.load(X + offs).to(tl.float32)
        b = tl.load(Bias + bias_idx).to(tl.float32)
        tl.store(Y + offs, v + b)


@triton.jit
def _add_bias_bwd(X, Bias, DY, DX, N: tl.constexpr, LAST_DIM: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        dx = tl.load(DY + offs).to(tl.float32)
        tl.store(DX + offs, dx)


class FusedAddBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        assert x.is_contiguous() and bias.is_contiguous()
        assert x.shape[-1] == bias.shape[0]
        ctx.save_for_backward(bias)
        y = torch.empty_like(x)
        N = x.numel()
        LAST_DIM = x.shape[-1]
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _add_bias_fwd[grid](x, bias, y, N, LAST_DIM, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        bias, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        N = dy.numel()
        LAST_DIM = dy.shape[-1]
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _add_bias_bwd[grid](dy, bias, dy, dx, N, LAST_DIM, BLOCK=BLOCK, num_warps=4)
        return dx, None


def kernel_fn(x, bias):
    return FusedAddBias.apply(x, bias)


def can_use_kernel(x, bias):
    return (x.is_cuda and bias.is_cuda and
            x.is_contiguous() and bias.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            x.shape[-1] == bias.shape[0])


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "bias": (4096,)},
    "vit_h":  {"x": (2, 2048, 5120), "bias": (5120,)},
    "small":  {"x": (8, 256, 1536), "bias": (1536,)},
}
