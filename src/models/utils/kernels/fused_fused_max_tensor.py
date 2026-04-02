"""Fused Max-Tensor kernel.
Pattern: y = max(a, b) elementwise
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl

def baseline_fn(a, b):
    return torch.max(a, b)

@triton.jit
def _max_tensor_fwd(A, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N: break
        av = tl.load(A + offs).to(tl.float32)
        bv = tl.load(B + offs).to(tl.float32)
        tl.store(Y + offs, av if av >= bv else bv)

@triton.jit
def _max_tensor_bwd(A, B, Y, DY, DA, DB, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N: break
        av = tl.load(A + offs).to(tl.float32)
        bv = tl.load(B + offs).to(tl.float32)
        dy = tl.load(DY + offs).to(tl.float32)
        amax = 1.0 if av >= bv else 0.0
        tl.store(DA + offs, dy * amax)
        tl.store(DB + offs, dy * (1.0 - amax))

class FusedMaxTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        assert a.is_contiguous() and b.is_contiguous()
        assert a.shape == b.shape
        ctx.save_for_backward(a, b)
        y = torch.empty_like(a)
        N = a.numel(); BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _max_tensor_fwd[grid](a, b, y, N, BLOCK=BLOCK, num_warps=4)
        return y
    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensors
        dy = dy.contiguous()
        da = torch.empty_like(a)
        db = torch.empty_like(a)
        N = a.numel(); BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _max_tensor_bwd[grid](a, b, a, dy, da, db, N, BLOCK=BLOCK, num_warps=4)
        return da, db

def kernel_fn(a, b):
    return FusedMaxTensor.apply(a, b)

def can_use_kernel(a, b):
    return a.is_cuda and b.is_cuda and a.is_contiguous() and b.is_contiguous() and a.shape == b.shape

SHAPES = {
    "vit_l":  {"a": (2, 1024, 4096), "b": (2, 1024, 4096)},
    "vit_h":  {"a": (2, 2048, 5120), "b": (2, 2048, 5120)},
    "small":  {"a": (8, 256, 1536), "b": (8, 256, 1536)},
}
