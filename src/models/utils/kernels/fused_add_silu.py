"""Fused Add-SiLU kernel.

Pattern: y = (a + b) * sigmoid(a + b)
Fuses: add + sigmoid + multiply into one elementwise pass.
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(a, b):
    return torch.nn.functional.silu(a + b)


# --- KERNEL ---
@triton.jit
def _add_silu_fwd(A, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        av = tl.load(A + offs).to(tl.float32)
        bv = tl.load(B + offs).to(tl.float32)
        s = av + bv
        sig = 1.0 / (1.0 + tl.exp(tl.minimum(-s, 20.0)))
        y = s * sig
        tl.store(Y + offs, y)


@triton.jit
def _add_silu_bwd(A, B, Y, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        av = tl.load(A + offs).to(tl.float32)
        bv = tl.load(B + offs).to(tl.float32)
        dy = tl.load(DY + offs).to(tl.float32)
        s = av + bv
        sig = 1.0 / (1.0 + tl.exp(tl.minimum(-s, 20.0)))
        dsig = sig * (1.0 - sig)
        dx = dy * (sig + s * dsig)
        tl.store(DX + offs, dx)


class FusedAddSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        assert a.is_contiguous() and b.is_contiguous()
        assert a.shape == b.shape
        ctx.save_for_backward(a, b)
        y = torch.empty_like(a)
        N = a.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _add_silu_fwd[grid](a, b, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensors
        dy = dy.contiguous()
        da = torch.empty_like(a)
        db = torch.empty_like(a)
        N = a.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _add_silu_bwd[grid](a, b, a, dy, da, N, BLOCK=BLOCK, num_warps=4)
        _add_silu_bwd[grid](a, b, a, dy, db, N, BLOCK=BLOCK, num_warps=4)
        return da, db


def kernel_fn(a, b):
    return FusedAddSiLU.apply(a, b)


def can_use_kernel(a, b):
    return (a.is_cuda and b.is_cuda and
            a.is_contiguous() and b.is_contiguous() and
            a.shape == b.shape and
            a.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"a": (2, 1024, 4096), "b": (2, 1024, 4096)},
    "vit_h":  {"a": (2, 2048, 5120), "b": (2, 2048, 5120)},
    "small":  {"a": (8, 256, 1536), "b": (8, 256, 1536)},
}
