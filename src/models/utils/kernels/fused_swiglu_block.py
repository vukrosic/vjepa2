"""Fused SiGLU Block kernel.

Pattern: y = SiLU(x @ W1 + b1) * (x @ W2 + b2)
Fuses: two linear projections + SiGLU activation + multiply into one kernel.
For MLLM FFN blocks with SwiGLU-style gating.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, w1, b1, w2, b2):
    hidden = torch.nn.functional.silu(torch.nn.functional.linear(x, w1, b1))
    gate = torch.nn.functional.linear(x, w2, b2)
    return hidden * gate


# --- KERNEL ---
@triton.jit
def _fused_swiglu_fwd(
    X, W1, B1, W2, B2, Y,
    B: tl.constexpr, M: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr,
    stride_xb, stride_xm,
    stride_w1n, stride_w1k,
    stride_yb, stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N1: tl.constexpr,
):
    """Fused SiGLU forward. One program per (batch, m) row."""
    bn = tl.program_id(0)
    bm = tl.program_id(1)
    b_offset = bn * stride_xb
    y_offset = bn * stride_yb

    # Compute gate = x @ W2 + b2 first (needed for output offset)
    # Then compute act = silu(x @ W1 + b1)
    # We'll do two matmul accumulations

    # accumulator for first projection
    acc1 = tl.zeros([BLOCK_N1], dtype=tl.float32)
    for k in range(M):
        x_val = tl.load(X + b_offset + bm * stride_xm + k).to(tl.float32)
        for n in range(BLOCK_N1):
            w1_val = tl.load(W1 + n * stride_w1n + k * stride_w1k).to(tl.float32)
            acc1[n] += x_val * w1_val

    # accumulator for second projection
    acc2 = tl.zeros([BLOCK_N1], dtype=tl.float32)
    for k in range(M):
        x_val = tl.load(X + b_offset + bm * stride_xm + k).to(tl.float32)
        for n in range(BLOCK_N1):
            w2_val = tl.load(W2 + n * stride_w1n + k * stride_w1k).to(tl.float32)
            acc2[n] += x_val * w2_val

    # Apply biases and silu
    for n in range(BLOCK_N1):
        b1_val = tl.load(B1 + n).to(tl.float32)
        b2_val = tl.load(B2 + n).to(tl.float32)
        p1 = acc1[n] + b1_val
        p2 = acc2[n] + b2_val
        sig = 1.0 / (1.0 + tl.exp(-p1))
        silu = p1 * sig
        out = silu * p2
        tl.store(Y + y_offset + bm * stride_ym + n, out)


class FusedSwigluBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        assert x.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()
        ctx.save_for_backward(x, w1, b1, w2, b2)
        B, M = x.shape
        N = w1.shape[0]
        y = torch.empty(B, M, N, dtype=x.dtype, device=x.device)
        grid = (B, M)
        _fused_swiglu_fwd[grid](
            x, w1, b1, w2, b2, y,
            B=B, M=M, N1=N, N2=N,
            x.stride(0), x.stride(1),
            w1.stride(0), w1.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=M, BLOCK_N1=triton.next_power_of_2(N),
            num_warps=8,
        )
        return y


def kernel_fn(x, w1, b1, w2, b2):
    return FusedSwigluBlock.apply(x, w1, b1, w2, b2)


def can_use_kernel(x, w1, b1, w2, b2):
    return (x.is_cuda and w1.is_cuda and w2.is_cuda and
            x.is_contiguous() and w1.is_contiguous() and w2.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            x.shape[-1] == w1.shape[1] == w2.shape[1])


SHAPES = {
    "vit_l": {"x": (2, 4096), "N": 4096, "M": 4096},
    "vit_h": {"x": (2, 5120), "N": 5120, "M": 5120},
    "small": {"x": (8, 1536), "N": 1536, "M": 1536},
}
