"""Fused Scale-Tensor kernel.

Pattern: y = x * scale
Fuses: scalar multiply into elementwise pass.
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, scale):
    return x * scale


# --- KERNEL ---
@triton.jit
def _scale_tensor_fwd(X, Y, N: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        v = tl.load(X + offs).to(tl.float32)
        tl.store(Y + offs, v * SCALE)


@triton.jit
def _scale_tensor_bwd(X, Y, DY, DX, N: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        dy = tl.load(DY + offs).to(tl.float32)
        tl.store(DX + offs, dy * SCALE)


class FusedScaleTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        assert x.is_contiguous()
        ctx.save_for_backward(x)
        ctx.scale = float(scale)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _scale_tensor_fwd[grid](x, y, N, float(scale), BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _scale_tensor_bwd[grid](x, x, dy, dx, N, ctx.scale, BLOCK=BLOCK, num_warps=4)
        return dx, None


def kernel_fn(x, scale):
    return FusedScaleTensor.apply(x, scale)


def can_use_kernel(x, scale):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
