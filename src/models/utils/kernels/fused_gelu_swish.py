"""Fused GELU * Sigmoid kernel.

Pattern: y = GELU(x) * sigmoid(other)
Fuses: GELU + sigmoid + multiply into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, other):
    return torch.nn.functional.gelu(x) * torch.sigmoid(other)


# --- KERNEL ---
@triton.jit
def _fused_gelu_swish_fwd(X, Other, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    o = tl.load(Other + offs, mask=mask).to(tl.float32)
    # GELU(x)
    k = 1.702
    sig_gelu = 1.0 / (1.0 + tl.exp(tl.minimum(-k * x, 20.0)))
    gelu = x * sig_gelu
    # sigmoid(other)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-o, 20.0)))
    y = gelu * sig
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_gelu_swish_bwd(X, Other, DY, DX, DO, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    o = tl.load(Other + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    k = 1.702
    sig_gelu = 1.0 / (1.0 + tl.exp(tl.minimum(-k * x, 20.0)))
    gelu = x * sig_gelu
    dsig_gelu = sig_gelu * (1.0 - sig_gelu)
    dgelu_dx = sig_gelu + k * x * dsig_gelu
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-o, 20.0)))
    dsig = sig * (1.0 - sig)
    dx = dy * dgelu_dx * sig
    do = dy * gelu * dsig
    tl.store(DX + offs, dx, mask=mask)
    tl.store(DO + offs, do, mask=mask)


class FusedGeluSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, other):
        assert x.is_contiguous() and other.is_contiguous()
        ctx.save_for_backward(x, other)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_swish_fwd[grid](x, other, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, other = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        do = torch.empty_like(other)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_swish_bwd[grid](x, other, dy, dx, do, N, BLOCK=BLOCK, num_warps=4)
        return dx, do


def kernel_fn(x, other):
    return FusedGeluSwish.apply(x, other)


def can_use_kernel(x, other):
    return (x.is_cuda and other.is_cuda and
            x.is_contiguous() and other.is_contiguous() and
            x.shape == other.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096), "other": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120), "other": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536), "other": (8, 256, 1536)},
}
