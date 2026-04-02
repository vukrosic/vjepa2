"""Fused Softmin v2 kernel.

Pattern: softmin(x, dim) = -softmax(x, dim)
Fuses: negation + softmax into one pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, dim=-1):
    return torch.nn.functional.softmin(x, dim=dim)


# --- KERNEL ---
@triton.jit
def _softmin_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    x_shifted = -x - tl.max(-x)
    e = tl.exp(x_shifted)
    sum_e = tl.sum(e)
    y = -e / sum_e
    tl.store(Y + offs, y, mask=mask)


class FusedSoftminV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=-1):
        assert x.is_contiguous()
        assert dim == -1
        y = torch.empty_like(x)
        N = x.shape[-1]
        BLOCK = 1024
        grid = (x.numel() // N,)
        _softmin_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("Softmin backward not yet implemented")


def kernel_fn(x, dim=-1):
    return FusedSoftminV2.apply(x, dim)


def can_use_kernel(x, dim=-1):
    return (x.is_cuda and
            x.is_contiguous() and
            dim == -1 and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
