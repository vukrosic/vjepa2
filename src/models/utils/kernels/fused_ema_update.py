"""Fused EMA (Exponential Moving Average) update kernel.

Source: V-JEPA 2 target encoder (EMA of student weights)
Pattern: target.data = momentum * target.data + (1 - momentum) * student.data
Fuses: The 3-term update (scale target, scale student, add) into 1 read + 1 write.
Frequency: Every training iteration, for all parameters (100M+ elements total).
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(target, student, momentum):
    """In-place EMA update: target = momentum * target + (1 - momentum) * student."""
    target.mul_(momentum).add_(student, alpha=1.0 - momentum)
    return target


# --- KERNEL ---
@triton.jit
def _ema_update_kernel(
    TARGET_ptr, STUDENT_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    momentum: tl.constexpr,
    one_minus_momentum: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    t = tl.load(TARGET_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(STUDENT_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    out = momentum * t + one_minus_momentum * s
    tl.store(TARGET_ptr + offs, out.to(t.dtype), mask=mask)


def kernel_fn(target, student, momentum):
    """In-place EMA update."""
    assert target.shape == student.shape
    assert target.is_contiguous() and student.is_contiguous()

    N = target.numel()
    BLOCK = 1024
    n_blocks = (N + BLOCK - 1) // BLOCK
    one_minus = 1.0 - momentum

    _ema_update_kernel[(n_blocks,)](
        target, student,
        N=N, BLOCK=BLOCK,
        momentum=momentum, one_minus_momentum=one_minus,
        num_warps=4,
    )
    return target


def can_use_kernel(target, student, momentum):
    return (target.is_cuda and student.is_cuda and
            target.is_contiguous() and student.is_contiguous() and
            target.shape == student.shape and
            target.dtype in (torch.float16, torch.bfloat16, torch.float32) and
            isinstance(momentum, float) and 0.0 < momentum < 1.0)


# Realistic shapes: ViT-L has ~300M params, test with typical layer sizes
SHAPES = {
    "attn_weight":  {"target": (1024, 1024), "student": (1024, 1024)},
    "mlp_weight":   {"target": (4096, 1024), "student": (4096, 1024)},
    "large_linear": {"target": (5120, 1280), "student": (5120, 1280)},
}
