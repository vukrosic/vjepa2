"""Fused gradient norm computation + clipping kernel.

Source: Training loop - torch.nn.utils.clip_grad_norm_
Pattern: Compute global L2 norm across all params, then scale each param's grad if norm > max_norm.
Fuses: The norm computation (partial sums per param) into a single kernel,
       then a second kernel applies the clip in-place.
Key win: Standard clip_grad_norm_ does 2 passes (one for norm, one for clip).
         This version reduces the norm in one fused pass per tensor, then clips.
Frequency: Every training iteration.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(grads, max_norm, norm_type=2.0):
    """
    grads: list of gradient tensors (already flat/contiguous).
    Returns: (clipped_grads, total_norm) where grads are clipped in-place.
    """
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
        norm_type
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)
    return grads, total_norm


# --- KERNEL ---
@triton.jit
def _partial_norm_sq_kernel(
    GRAD_ptr, PARTIAL_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute partial sum of squared gradient elements."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    g = tl.load(GRAD_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sq = g * g
    block_sum = tl.sum(tl.where(mask, sq, 0.0), axis=0)
    tl.store(PARTIAL_ptr + pid, block_sum)


@triton.jit
def _clip_grad_kernel(
    GRAD_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    clip_coef: tl.constexpr,
):
    """Scale gradient by clip_coef in-place."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    g = tl.load(GRAD_ptr + offs, mask=mask, other=0.0)
    g_scaled = g.to(tl.float32) * clip_coef
    tl.store(GRAD_ptr + offs, g_scaled.to(g.dtype), mask=mask)


def _compute_grad_norm(grads, BLOCK=1024):
    """Compute L2 norm across all gradients using Triton partial sums."""
    partial_sums = []
    for g in grads:
        N = g.numel()
        n_blocks = (N + BLOCK - 1) // BLOCK
        partials = torch.empty(n_blocks, dtype=torch.float32, device=g.device)
        _partial_norm_sq_kernel[(n_blocks,)](
            g.contiguous(), partials, N=N, BLOCK=BLOCK, num_warps=4
        )
        partial_sums.append(partials)
    total_sq = torch.cat(partial_sums).sum()
    return total_sq.sqrt()


def kernel_fn(grads, max_norm, norm_type=2.0):
    """Fused gradient norm + clipping."""
    assert norm_type == 2.0, "Kernel only supports L2 norm"
    BLOCK = 1024

    total_norm = _compute_grad_norm(grads, BLOCK)
    clip_coef = (max_norm / (total_norm.item() + 1e-6))

    if clip_coef < 1.0:
        for g in grads:
            g_c = g.contiguous()
            N = g_c.numel()
            n_blocks = (N + BLOCK - 1) // BLOCK
            _clip_grad_kernel[(n_blocks,)](
                g_c, N=N, BLOCK=BLOCK, clip_coef=clip_coef, num_warps=4
            )
            if not g.is_contiguous():
                g.copy_(g_c)

    return grads, total_norm


def can_use_kernel(grads, max_norm, norm_type=2.0):
    if norm_type != 2.0:
        return False
    if not all(g.is_cuda for g in grads):
        return False
    return True


# Shapes represent typical gradient tensors in ViT-L
SHAPES = {
    "attn_weight":  {"grad": (1024, 1024)},
    "mlp_weight":   {"grad": (4096, 1024)},
    "vit_h_weight": {"grad": (5120, 1280)},
}
