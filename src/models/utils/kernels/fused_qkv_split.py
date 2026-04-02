"""Fused QKV split + transpose kernel.

Source: src/models/utils/modules.py (Attention.forward, RoPEAttention.forward)
Pattern: qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4); q, k, v = qkv[0], qkv[1], qkv[2]
Fuses: reshape + permute + unbind into a single copy that writes q, k, v into
       three separate contiguous buffers [B, H, N, D] in one kernel pass.
The PyTorch path creates non-contiguous views requiring a later .contiguous() call.
Frequency: Every attention block.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(qkv_linear, B, N, H, D):
    """
    qkv_linear: [B, N, 3*H*D] output of the QKV linear layer
    Returns: q, k, v each [B, H, N, D] contiguous
    """
    qkv = qkv_linear.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()
    return q, k, v


# --- KERNEL ---
@triton.jit
def _qkv_split_kernel(
    QKV_ptr,   # [B, N, 3, H, D] flattened input
    Q_ptr,     # [B, H, N, D] output
    K_ptr,     # [B, H, N, D] output
    V_ptr,     # [B, H, N, D] output
    B, N, H,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Program grid: (B * N * H,)
    Each program handles one (b, n, h) slot and copies D elements for q, k, v.
    Source: qkv[b, n, qkv_idx, h, d] = QKV_ptr[b*N*3*H*D + n*3*H*D + qkv_idx*H*D + h*D + d]
    Dest:   q[b, h, n, d]             = Q_ptr  [b*H*N*D + h*N*D + n*D + d]
    """
    pid = tl.program_id(0)
    # Decode (b, n, h) from pid
    h_idx = pid % H
    n_idx = (pid // H) % N
    b_idx = pid // (H * N)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Source base: b*N*3*H*D + n*3*H*D + qkv*H*D + h*D
    src_base = b_idx * N * 3 * H * D + n_idx * 3 * H * D + h_idx * D

    # Dest base: b*H*N*D + h*N*D + n*D
    dst_base = b_idx * H * N * D + h_idx * N * D + n_idx * D

    # Load and store Q (qkv_idx=0)
    q_src = src_base + 0 * H * D
    q = tl.load(QKV_ptr + q_src + offs_d, mask=mask_d)
    tl.store(Q_ptr + dst_base + offs_d, q, mask=mask_d)

    # Load and store K (qkv_idx=1)
    k_src = src_base + 1 * H * D
    k = tl.load(QKV_ptr + k_src + offs_d, mask=mask_d)
    tl.store(K_ptr + dst_base + offs_d, k, mask=mask_d)

    # Load and store V (qkv_idx=2)
    v_src = src_base + 2 * H * D
    v = tl.load(QKV_ptr + v_src + offs_d, mask=mask_d)
    tl.store(V_ptr + dst_base + offs_d, v, mask=mask_d)


def kernel_fn(qkv_linear, B, N, H, D):
    """
    qkv_linear: [B, N, 3*H*D] output of the QKV linear layer
    Returns: q, k, v each [B, H, N, D] contiguous
    """
    qkv_c = qkv_linear.contiguous()
    # Reshape to [B, N, 3, H, D] without permute — stays contiguous
    qkv_reshaped = qkv_c.view(B, N, 3, H, D)

    q = torch.empty(B, H, N, D, dtype=qkv_c.dtype, device=qkv_c.device)
    k = torch.empty(B, H, N, D, dtype=qkv_c.dtype, device=qkv_c.device)
    v = torch.empty(B, H, N, D, dtype=qkv_c.dtype, device=qkv_c.device)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (B * N * H,)
    _qkv_split_kernel[grid](
        qkv_reshaped, q, k, v,
        B, N, H, D=D, BLOCK_D=BLOCK_D,
        num_warps=min(8, max(1, BLOCK_D // 32)),
    )
    return q, k, v


def can_use_kernel(qkv_linear, B, N, H, D):
    if not qkv_linear.is_cuda:
        return False
    D_int = int(D)
    if D_int > 256:  # BLOCK_D limit for constexpr
        return False
    if qkv_linear.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


# ViT-L: dim=1024, num_heads=16, head_dim=64
# ViT-H: dim=1280, num_heads=16, head_dim=80 — head_dim=80 > 64, use BLOCK_D=128
SHAPES = {
    "vit_l": {"qkv_linear": (2, 1024, 3072), "B": 2, "N": 1024, "H": 16, "D": 64},
    "vit_s": {"qkv_linear": (2, 256, 1152), "B": 2, "N": 256, "H": 12, "D": 64},
    "vit_b": {"qkv_linear": (4, 512, 2304), "B": 4, "N": 512, "H": 12, "D": 64},
}
