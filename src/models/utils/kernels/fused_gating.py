"""Fused gating kernel.

Pattern: y = gate * input (used in feed-forward gating like SwiGLU)
Fuses: sigmoid(gate) * input or gate * input into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(gate, input):
    return torch.nn.functional.silu(gate) * input


@triton.jit
def _gating_fwd(GATE, INPUT, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
    i = tl.load(INPUT + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    y = sig * i
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _gating_bwd(GATE, INPUT, DY, DGATE, DINPUT, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
    i = tl.load(INPUT + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    dsig = sig * (1.0 - sig)
    # d(sig*i)/dg = dsig * i; d(sig*i)/di = sig
    tl.store(DGATE + offs, dy * dsig * i, mask=mask)
    tl.store(DINPUT + offs, dy * sig, mask=mask)


class FusedGating(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, input):
        gate = gate.contiguous()
        input = input.contiguous()
        y = torch.empty_like(gate)
        N = gate.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _gating_fwd[(n_blocks,)](gate, input, y, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(gate, input)
        return y

    @staticmethod
    def backward(ctx, dy):
        gate, input = ctx.saved_tensors
        dgate = torch.empty_like(gate)
        dinput = torch.empty_like(gate)
        N = dy.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _gating_bwd[(n_blocks,)](gate, input, dy, dgate, dinput, N, BLOCK=BLOCK, num_warps=4)
        return dgate, dinput


def kernel_fn(gate, input):
    return FusedGating.apply(gate, input)


def can_use_kernel(gate, input):
    return (gate.is_cuda and gate.is_contiguous() and input.is_contiguous() and
            gate.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"gate": (2, 1024, 8192), "input": (2, 1024, 8192)},
    "vit_h":  {"gate": (2, 2048, 10240), "input": (2, 2048, 10240)},
    "small":  {"gate": (8, 256, 3072), "input": (8, 256, 3072)},
}
