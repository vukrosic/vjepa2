"""Fused Mish-Gate activation kernel.

Pattern: Mish-GLU = mish(x) * gate(x)
Fuses: softplus + tanh + multiply + gate multiply into one pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x, gate):
    return torch.nn.functional.mish(x) * gate


# --- KERNEL ---
@triton.jit
def _fused_mish_gate_fwd(X, GATE, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    g = tl.load(GATE + offs, mask=mask).to(tl.float32)
    # Mish: x * tanh(softplus(x)) = x * tanh(log(exp(x) + 1))
    sp = tl.log(tl.exp(tl.minimum(x, 20.0)) + 1.0)
    e2sp = tl.exp(tl.minimum(2.0 * sp, 40.0))
    tanh_sp = (e2sp - 1.0) / (e2sp + 1.0)
    mish = x * tanh_sp
    y = mish * g
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _fused_mish_gate_bwd(X, GATE, DY, DX, DGATE, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    g = tl.load(GATE + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    # Forward intermediates
    sp = tl.log(tl.exp(tl.minimum(x, 20.0)) + 1.0)
    e2sp = tl.exp(tl.minimum(2.0 * sp, 40.0))
    tanh_sp = (e2sp - 1.0) / (e2sp + 1.0)
    mish = x * tanh_sp
    # d(mish)/dx = tanh(sp) + x * (1 - tanh^2(sp)) * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(tl.minimum(-x, 20.0)))
    dtanh_dsp = 1.0 - tanh_sp * tanh_sp
    dmish_dx = tanh_sp + x * dtanh_dsp * sig
    # d(out)/dx = dmish_dx * gate
    dx = dy * dmish_dx * g
    # d(out)/dgate = mish
    dgate = dy * mish
    tl.store(DX + offs, dx, mask=mask)
    tl.store(DGATE + offs, dgate, mask=mask)


class FusedMishGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate):
        assert x.is_contiguous() and gate.is_contiguous()
        ctx.save_for_backward(x, gate)
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_mish_gate_fwd[grid](x, gate, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, gate = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        dgate = torch.empty_like(gate)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_mish_gate_bwd[grid](x, gate, dy, dx, dgate, N, BLOCK=BLOCK, num_warps=4)
        return dx, dgate


def kernel_fn(x, gate):
    return FusedMishGate.apply(x, gate)


def can_use_kernel(x, gate):
    return (x.is_cuda and gate.is_cuda and
            x.is_contiguous() and gate.is_contiguous() and
            x.shape == gate.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096), "gate": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120), "gate": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536), "gate": (8, 256, 1536)},
}
