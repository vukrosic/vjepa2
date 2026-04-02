"""Fused SiLU * x2 kernel for SwiGLU MLP.

Source: src/models/utils/modules.py:178-182 (SwiGLUFFN.forward)
Pattern: hidden = F.silu(fc1(x)) * fc2(x)
Fuses: SiLU activation + elementwise multiply into one read/write pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x1, x2):
    return torch.nn.functional.silu(x1) * x2


# --- KERNEL ---
@triton.jit
def _fused_silu_mul_fwd(X1, X2, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x1 = tl.load(X1 + offs, mask=mask).to(tl.float32)
    x2 = tl.load(X2 + offs, mask=mask).to(tl.float32)
    # SiLU: x * sigmoid(x)
    silu = x1 * tl.sigmoid(x1)
    y = silu * x2
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_silu_mul_bwd(X1, X2, DY, DX1, DX2, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x1 = tl.load(X1 + offs, mask=mask).to(tl.float32)
    x2 = tl.load(X2 + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    # Forward: y = silu(x1) * x2 = x1 * sigmoid(x1) * x2
    sig = tl.sigmoid(x1)
    silu = x1 * sig
    # d(silu)/dx1 = sigmoid(x1) + x1 * sigmoid(x1) * (1 - sigmoid(x1))
    #             = sigmoid(x1) * (1 + x1 * (1 - sigmoid(x1)))
    dsilu_dx1 = sig * (1.0 + x1 * (1.0 - sig))
    # dy/dx1 = dsilu_dx1 * x2
    dx1 = dy * dsilu_dx1 * x2
    # dy/dx2 = silu(x1)
    dx2 = dy * silu
    tl.store(DX1 + offs, dx1, mask=mask)
    tl.store(DX2 + offs, dx2, mask=mask)


class FusedSiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        assert x1.is_contiguous() and x2.is_contiguous()
        ctx.save_for_backward(x1, x2)
        y = torch.empty_like(x1)
        N = x1.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_silu_mul_fwd[grid](x1, x2, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x1, x2 = ctx.saved_tensors
        dy = dy.contiguous()
        dx1 = torch.empty_like(x1)
        dx2 = torch.empty_like(x2)
        N = x1.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_silu_mul_bwd[grid](x1, x2, dy, dx1, dx2, N, BLOCK=BLOCK, num_warps=4)
        return dx1, dx2


def kernel_fn(x1, x2):
    return FusedSiLUMul.apply(x1, x2)


def can_use_kernel(x1, x2):
    return (x1.is_cuda and x2.is_cuda and
            x1.is_contiguous() and x2.is_contiguous() and
            x1.shape == x2.shape and
            x1.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x1": (2, 1024, 2730), "x2": (2, 1024, 2730)},   # ViT-L SwiGLU hidden
    "vit_h": {"x1": (2, 4096, 3416), "x2": (2, 4096, 3416)},   # ViT-H SwiGLU hidden
    "small":  {"x1": (8, 256, 1024), "x2": (8, 256, 1024)},     # small batch
}
