"""Fused SwiGLU activation kernel.

Pattern: SwiGLU(x) = SiLU(W1*x) * (W2*x)
Fuses: W1 projection + SiLU + W2 projection + elementwise multiply into one pass.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x, w1, w2):
    """SwiGLU: SiLU(W1*x) * (W2*x)"""
    x1 = F.linear(x, w1)
    x2 = F.linear(x, w2)
    return F.silu(x1) * x2


# --- KERNEL ---
@triton.jit
def _fused_swiglu_fwd(
    X, W1, W2, Y,
    B: tl.constexpr, D: tl.constexpr, D2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per batch element. BLOCK_D tiles the output dimension."""
    pid_b = tl.program_id(0)
    X_base = pid_b * D
    Y_base = pid_b * D2

    for j in range(0, D2, BLOCK_D):
        offs_out = tl.arange(0, BLOCK_D)
        mask_out = offs_out < D2

        # Compute gate = SiLU(W1x)
        gate_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(D):
            x_val = tl.load(X + X_base + i).to(tl.float32)
            w1_col = tl.load(W1 + i * D + offs_out).to(tl.float32)
            gate_acc = gate_acc + x_val * w1_col

        # SiLU: x * sigmoid(x)
        sig = 1.0 / (1.0 + tl.exp(tl.minimum(-gate_acc, 20.0)))
        silu_gate = gate_acc * sig

        # Compute val = W2x
        val_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(D):
            x_val = tl.load(X + X_base + i).to(tl.float32)
            w2_col = tl.load(W2 + i * D2 + offs_out).to(tl.float32)
            val_acc = val_acc + x_val * w2_col

        y = silu_gate * val_acc
        tl.store(Y + Y_base + offs_out, y, mask=mask_out)


class FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2):
        assert x.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()
        B, D = x.shape
        D2 = w2.shape[1]
        y = torch.empty(B, D2, dtype=x.dtype, device=x.device)
        BLOCK_D = triton.next_power_of_2(D2)
        BLOCK_D = min(BLOCK_D, 4096)
        _fused_swiglu_fwd[(B,)](
            x, w1, w2, y, B, D, D2, BLOCK_D=BLOCK_D, num_warps=4,
        )
        ctx.save_for_backward(x, w1, w2)
        ctx.B = B
        ctx.D = D
        ctx.D2 = D2
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w1, w2 = ctx.saved_tensors
        # Delegate to autograd for complex multi-output gradient
        grad = torch.autograd.grad(
            F.silu(F.linear(x, w1)) * F.linear(x, w2),
            (x, w1, w2),
            grad_outputs=(dy.contiguous(),),
        )
        return grad


def kernel_fn(x, w1, w2):
    return FusedSwiGLU.apply(x, w1, w2)


def can_use_kernel(x, w1, w2):
    if not (x.is_cuda and w1.is_cuda and w2.is_cuda):
        return False
    if not (x.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()):
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if w1.shape[0] != x.shape[-1] or w2.shape[0] != x.shape[-1]:
        return False
    if w1.shape[1] != w2.shape[1]:
        return False
    return True


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096), "w1": (4096, 4096), "w2": (4096, 4096)},
    "vit_h": {"x": (2, 1024, 5120), "w1": (5120, 5120), "w2": (5120, 5120)},
    "small": {"x": (8, 256, 1536), "w1": (1536, 1536), "w2": (1536, 1536)},
}
