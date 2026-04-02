"""Fused SoftArgmax kernel.

Pattern: y = sum(i * softmax(x)_i) = argmax as differentiable proxy
Fuses: softmax + weighted sum into one pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, temp=1.0):
    # Soft argmax: weighted average of indices
    e = torch.exp(x / temp)
    softmax = e / e.sum(dim=-1, keepdim=True)
    indices = torch.arange(x.shape[-1], device=x.device, dtype=softmax.dtype)
    return (softmax * indices).sum(dim=-1)


# --- KERNEL ---
@triton.jit
def _soft_argmax_fwd(X, Y, N: tl.constexpr, TEMP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    x_shifted = x / TEMP - tl.max(x / TEMP)
    e = tl.exp(x_shifted)
    sum_e = tl.sum(e)
    prob = e / sum_e
    idx = offs.astype(tl.float32)
    y = tl.sum(prob * idx)
    tl.store(Y + pid, y)


class FusedSoftArgmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, temp=1.0):
        assert x.is_contiguous()
        y = torch.empty(x.shape[:-1], dtype=x.dtype, device=x.device)
        N = x.shape[-1]
        BLOCK = 1024
        grid = (x.numel() // N,)
        _soft_argmax_fwd[grid](x, y, N, temp, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("SoftArgmax backward not yet implemented")


def kernel_fn(x, temp=1.0):
    return FusedSoftArgmax.apply(x, temp)


def can_use_kernel(x, temp=1.0):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "temp": 1.0},
    "vit_h":  {"x": (2, 2048, 5120), "temp": 1.0},
    "small":  {"x": (8, 256, 1536), "temp": 1.0},
}
