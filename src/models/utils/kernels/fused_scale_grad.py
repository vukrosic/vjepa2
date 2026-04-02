"""Fused gradient scaling kernel.

For mixed precision training: gradients are stored in fp32 but computed in fp16.
This kernel handles scaling gradients by a factor (e.g., for loss scaling in AMP).
Pattern: grad = grad * scale; grad = grad.clamp(-max_val, max_val)
Fuses: multiply + clamp into one kernel pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(grad, scale, max_val):
    """Scale gradients and clamp in one in-place operation."""
    return (grad * scale).clamp_(-max_val, max_val)


# --- KERNEL ---
@triton.jit
def _scale_clamp_kernel(
    GRAD, OUT,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    scale: tl.constexpr,
    max_val: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(GRAD + offs, mask=mask, other=0.0).to(tl.float32)
    g_scaled = g * scale
    g_clamped = tl.clamp(g_scaled, -max_val, max_val)
    tl.store(OUT + offs, g_clamped.to(g.dtype), mask=mask)


def kernel_fn(grad, scale, max_val):
    """Scale and clamp gradients."""
    N = grad.numel()
    BLOCK = 1024
    n_blocks = (N + BLOCK - 1) // BLOCK
    out = torch.empty_like(grad)
    _scale_clamp_kernel[(n_blocks,)](
        grad, out, N, BLOCK=BLOCK,
        scale=scale, max_val=max_val,
        num_warps=4,
    )
    return out


def can_use_kernel(grad, scale, max_val):
    return (grad.is_cuda and grad.is_contiguous() and
            grad.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l_grad":   {"grad": (2, 1024, 1024),},
    "vit_h_grad":   {"grad": (2, 2048, 1280),},
    "small_grad":   {"grad": (8, 256, 384),},
}
