"""Fused ELU + SiLU fusion kernel.

Pattern: y = ELU(x) * SiLU(gate)
Common in SwiGLU-like architectures.
Fuses: elu + silu + multiply into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, gate):
    return torch.nn.functional.elu(x) * torch.nn.functional.silu(gate)


# --- KERNEL ---
@triton.jit
def _fused_elu_silu_fwd(X, GATE, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    g = tl.load(GATE + offs, mask=mask).to(tl.float32)
    elu_x = tl.where(x >= 0, x, tl.exp(tl.minimum(x, 40.0)) - 1.0)
    silu_g = g * (1.0 / (1.0 + tl.exp(-g)))
    y = elu_x * silu_g
    tl.store(Y + offs, y, mask=mask)


class FusedELUSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate):
        assert x.is_contiguous() and gate.is_contiguous()
        ctx.save_for_backward(x, gate)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_elu_silu_fwd[grid](x, gate, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("ELUSiLU backward not yet implemented")


def kernel_fn(x, gate):
    return FusedELUSiLU.apply(x, gate)


def can_use_kernel(x, gate):
    return (x.is_cuda and gate.is_cuda and
            x.is_contiguous() and gate.is_contiguous() and
            x.shape == gate.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096), "gate": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120), "gate": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536), "gate": (8, 256, 1536)},
}
