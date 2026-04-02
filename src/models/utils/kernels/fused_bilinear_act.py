"""Fused bilinear (GLU) activation: out = x1 * sigmoid(x2).

Replaces the gated linear unit (GLU) pattern used in some MLP variants:
  out = (W1 @ x) * sigmoid(W2 @ x)

Fuses: sigmoid(x2) + elementwise multiply into one read/write pass,
avoiding materializing sigmoid(x2) as a separate tensor.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x1, x2):
    """x1, x2: [B, N, D] — outputs of two linear layers."""
    return x1 * torch.sigmoid(x2)


# --- KERNELS ---
@triton.jit
def _fused_bilinear_act_fwd(X1, X2, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x1 = tl.load(X1 + offs, mask=mask).to(tl.float32)
    x2 = tl.load(X2 + offs, mask=mask).to(tl.float32)
    sig = tl.sigmoid(x2)
    y = x1 * sig
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_bilinear_act_bwd(X1, X2, DY, DX1, DX2, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x1 = tl.load(X1 + offs, mask=mask).to(tl.float32)
    x2 = tl.load(X2 + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    sig = tl.sigmoid(x2)
    # dy/dx1 = sigmoid(x2)
    dx1 = dy * sig
    # dy/dx2 = x1 * sigmoid(x2) * (1 - sigmoid(x2))
    dx2 = dy * x1 * sig * (1.0 - sig)
    tl.store(DX1 + offs, dx1, mask=mask)
    tl.store(DX2 + offs, dx2, mask=mask)


class FusedBilinearAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        assert x1.is_contiguous() and x2.is_contiguous()
        ctx.save_for_backward(x1, x2)
        y = torch.empty_like(x1)
        N = x1.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_bilinear_act_fwd[grid](x1, x2, y, N=N, BLOCK=BLOCK, num_warps=4)
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
        _fused_bilinear_act_bwd[grid](x1, x2, dy, dx1, dx2, N=N, BLOCK=BLOCK, num_warps=4)
        return dx1, dx2


def kernel_fn(x1, x2):
    return FusedBilinearAct.apply(x1, x2)


def can_use_kernel(x1, x2):
    return (
        x1.is_cuda
        and x2.is_cuda
        and x1.is_contiguous()
        and x2.is_contiguous()
        and x1.shape == x2.shape
        and x1.dtype in (torch.float16, torch.float32, torch.bfloat16)
    )


SHAPES = {
    "vit_l": {"x1": (2, 1024, 2730), "x2": (2, 1024, 2730)},
    "vit_h": {"x1": (2, 2048, 3416), "x2": (2, 2048, 3416)},
    "small": {"x1": (8, 256, 1024),  "x2": (8, 256, 1024)},
}
