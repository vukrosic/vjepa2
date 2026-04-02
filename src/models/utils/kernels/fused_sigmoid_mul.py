"""Fused Sigmoid * Multiply kernel.

Pattern: y = sigmoid(x_gate) * y_value
Fuses: sigmoid activation + elementwise multiply into one pass.
Used in highway networks, LSTMs, and gate mechanisms in transformer FFNs.
Uses pure scalar loads for elementwise computation.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x_gate, y_value):
    return torch.sigmoid(x_gate) * y_value


# --- KERNEL ---
@triton.jit
def _fused_sigmoid_mul_fwd(X, Y, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.load(Y + offs, mask=mask).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    out = sig * y
    tl.store(Out + offs, out, mask=mask)


@triton.jit
def _fused_sigmoid_mul_bwd(X, Y, DOut, DX, DY, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.load(Y + offs, mask=mask).to(tl.float32)
    dout = tl.load(DOut + offs, mask=mask).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    # d(sigmoid) = sig * (1 - sig)
    dsig = sig * (1.0 - sig)
    # d(sigmoid * y)/dx = dsig * y
    dx = dout * dsig * y
    # d(sigmoid * y)/dy = sigmoid(x)
    dy = dout * sig
    tl.store(DX + offs, dx, mask=mask)
    tl.store(DY + offs, dy, mask=mask)


class FusedSigmoidMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_gate, y_value):
        assert x_gate.is_contiguous() and y_value.is_contiguous()
        ctx.save_for_backward(x_gate, y_value)
        y = torch.empty_like(x_gate)
        N = x_gate.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_sigmoid_mul_fwd[grid](x_gate, y_value, y, N, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x_gate, y_value = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x_gate)
        dy_val = torch.empty_like(y_value)
        N = x_gate.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_sigmoid_mul_bwd[grid](x_gate, y_value, dy, dx, dy_val, N, BLOCK=BLOCK, num_warps=4)
        return dx, dy_val


def kernel_fn(x_gate, y_value):
    return FusedSigmoidMul.apply(x_gate, y_value)


def can_use_kernel(x_gate, y_value):
    return (x_gate.is_cuda and y_value.is_cuda and
            x_gate.is_contiguous() and y_value.is_contiguous() and
            x_gate.shape == y_value.shape and
            x_gate.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 2730), "y": (2, 1024, 2730)},
    "vit_h": {"x": (2, 2048, 3416), "y": (2, 2048, 3416)},
    "small": {"x": (8, 256, 1024), "y": (8, 256, 1024)},
}
