"""Fused GLU (Gated Linear Unit) kernel.

Pattern: y = gate(x2) * x1 where x is split in half
Fuses: split + sigmoid/mul into one pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return torch.nn.functional.glu(x)


# --- KERNEL ---
@triton.jit
def _fused_glu_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    half = N // 2
    a = tl.load(X + offs, mask=(offs < half), other=0.0).to(tl.float32)
    b = tl.load(X + half + offs, mask=((half + offs) < N), other=0.0).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-b))
    y = a * gate
    tl.store(Y + offs, y, mask=mask)


class FusedGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        assert x.shape[-1] % 2 == 0
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_glu_fwd[grid](x, y, x.shape[-1], BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("GLU backward not yet implemented")


def kernel_fn(x):
    return FusedGLU.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.shape[-1] % 2 == 0 and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
