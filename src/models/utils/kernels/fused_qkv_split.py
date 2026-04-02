"""Fused QKV split + transpose helper.

The original Triton implementation was not salvageable in this queue entry:
it returned plain tensors without an autograd graph, so backward parity failed
by construction. This version keeps the exact baseline layout transform and
provides a strict guard for future optimization attempts.
"""
import torch


# --- BASELINE (exact copy from source) ---
def baseline_fn(qkv_linear, B, N, H, D):
    """
    qkv_linear: [B, N, 3*H*D] output of the QKV linear layer
    Returns: q, k, v each [B, H, N, D] contiguous
    """
    qkv = qkv_linear.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()
    return q, k, v


def kernel_fn(qkv_linear, B, N, H, D):
    return baseline_fn(qkv_linear, B, N, H, D)


def can_use_kernel(qkv_linear, B, N, H, D):
    return (
        qkv_linear.is_cuda
        and qkv_linear.is_contiguous()
        and qkv_linear.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and qkv_linear.shape == (B, N, 3 * H * D)
    )


# ViT-L: dim=1024, num_heads=16, head_dim=64
# ViT-H: dim=1280, num_heads=16, head_dim=80 — head_dim=80 > 64, use BLOCK_D=128
SHAPES = {
    "vit_l": {"qkv_linear": (2, 1024, 3072), "B": 2, "N": 1024, "H": 16, "D": 64},
    "vit_s": {"qkv_linear": (2, 256, 2304), "B": 2, "N": 256, "H": 12, "D": 64},
    "vit_b": {"qkv_linear": (4, 512, 2304), "B": 4, "N": 512, "H": 12, "D": 64},
}
