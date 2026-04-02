"""Fused attention output transpose + reshape kernel.

Source: src/models/utils/modules.py:517 (Attention.forward)
Pattern: x = x.transpose(1, 2).reshape(B, N, C)
         where x: [B, H, N, D] -> [B, N, C] with C = H*D
Fuses: transpose + reshape into a single contiguous copy kernel.
The PyTorch path creates non-contiguous tensors requiring extra memory ops.
Frequency: Every attention block.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, B, N, H, D):
    """x: [B, H, N, D] -> [B, N, H*D] contiguous."""
    return x.transpose(1, 2).reshape(B, N, H * D).contiguous()


# --- KERNEL ---
@triton.jit
def _attn_transpose_kernel(
    X_ptr,   # [B, H, N, D] input
    Y_ptr,   # [B, N, H*D] output
    B, N, H,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (B * N * H,)
    Each program handles one (b, n, h) output slot and copies D elements.
    Source: x[b, h, n, d]    = X_ptr[b*H*N*D + h*N*D + n*D + d]
    Dest:   y[b, n, h*D + d] = Y_ptr[b*N*H*D + n*H*D + h*D + d]
    """
    pid = tl.program_id(0)
    h_idx = pid % H
    n_idx = (pid // H) % N
    b_idx = pid // (H * N)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    src_off = b_idx * H * N * D + h_idx * N * D + n_idx * D
    dst_off = b_idx * N * H * D + n_idx * H * D + h_idx * D

    x = tl.load(X_ptr + src_off + offs_d, mask=mask_d)
    tl.store(Y_ptr + dst_off + offs_d, x, mask=mask_d)


def kernel_fn(x, B, N, H, D):
    """x: [B, H, N, D] -> [B, N, H*D] contiguous."""
    x_c = x.contiguous()
    y = torch.empty(B, N, H * D, dtype=x_c.dtype, device=x_c.device)
    BLOCK_D = triton.next_power_of_2(D)
    _attn_transpose_kernel[(B * N * H,)](
        x_c, y, B, N, H, D=D, BLOCK_D=BLOCK_D,
        num_warps=min(8, max(1, BLOCK_D // 32)),
    )
    return y


def can_use_kernel(x, B, N, H, D):
    if not x.is_cuda:
        return False
    if D > 256:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


# ViT-L: H=16, D=64; ViT-H: H=16, D=80
SHAPES = {
    "vit_l": {"x": (2, 16, 1024, 64), "B": 2, "N": 1024, "H": 16, "D": 64},
    "vit_h": {"x": (2, 16, 2048, 80), "B": 2, "N": 2048, "H": 16, "D": 80},
    "small": {"x": (4, 12, 256, 64), "B": 4, "N": 256, "H": 12, "D": 64},
}
