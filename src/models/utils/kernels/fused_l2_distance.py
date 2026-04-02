"""Fused L2 distance kernel for contrastive learning.

Pattern: dist = torch.norm(a - b, dim=-1)
Fuses: subtract + sqr + sum + sqrt in one read/write pass, no materialization of (a - b).
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(a, b):
    return torch.norm(a - b, dim=-1)


# --- KERNEL ---
@triton.jit
def _fused_l2_dist_fwd(A, B, Y, N: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    sq_sum = 0.0
    for d in range(0, D, BLOCK_D):
        a = tl.load(A + pid * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        b = tl.load(B + pid * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        diff = a - b
        sq_sum += diff * diff
        offs_d += BLOCK_D
        mask_d = offs_d < D
    dist = tl.sqrt(sq_sum)
    tl.store(Y + pid, dist)


def kernel_fn(a, b):
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape == b.shape
    B, D = a.shape
    y = torch.empty(B, dtype=a.dtype, device=a.device)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B,)
    _fused_l2_dist_fwd[grid](a, b, y, B, D, BLOCK_D=BLOCK_D, num_warps=4)
    return y


def can_use_kernel(a, b):
    return (a.is_cuda and b.is_cuda and
            a.is_contiguous() and b.is_contiguous() and
            a.shape == b.shape and
            a.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "small":  {"a": (256, 384),  "b": (256, 384)},
    "medium": {"a": (1024, 768), "b": (1024, 768)},
    "large":  {"a": (4096, 1024), "b": (4096, 1024)},
}
