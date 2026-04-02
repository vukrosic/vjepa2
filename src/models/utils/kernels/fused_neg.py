"""Fused Negation kernel.

Pattern: y = -x
Fuses: negation into one elementwise pass.
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return -x


# --- KERNEL ---
@triton.jit
def _neg_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        v = tl.load(X + offs).to(tl.float32)
        tl.store(Y + offs, -v)


@triton.jit
def _neg_bwd(DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        dy = tl.load(DY + offs).to(tl.float32)
        tl.store(DX + offs, -dy)


class FusedNeg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _neg_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        N = dy.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _neg_bwd[grid](dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedNeg.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
