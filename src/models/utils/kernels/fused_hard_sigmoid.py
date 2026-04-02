"""Fused HardSigmoid activation kernel.

Pattern: clamp(x + 3, 0, 6) / 6
Fuses: add + clamp + divide into one elementwise pass.
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x):
    return torch.nn.functional.hardsigmoid(x)


# --- KERNEL ---
@triton.jit
def _hard_sigmoid_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        v = tl.load(X + offs).to(tl.float32)
        y = tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0) / 6.0
        tl.store(Y + offs, y)


@triton.jit
def _hard_sigmoid_bwd(X, Y, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        x = tl.load(X + offs).to(tl.float32)
        dy = tl.load(DY + offs).to(tl.float32)
        inside = (x > -3.0) and (x < 3.0)
        dx = dy * (1.0 / 6.0) if inside else 0.0
        tl.store(DX + offs, dx)


class FusedHardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _hard_sigmoid_fwd[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _hard_sigmoid_bwd[grid](x, x, dy, dx, N, BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return FusedHardSigmoid.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
