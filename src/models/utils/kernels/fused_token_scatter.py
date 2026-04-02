"""Fused token scatter kernel.

Scatters M unmasked tokens back into a full token grid of size total_tokens.
This is the reverse of gather — used in the V-JEPA 2 predictor after
masked token prediction to reconstruct the full feature map.

Baseline uses torch.scatter_ with zero-fill. The Triton kernel does one
program per (B, M) token and copies D features in a single vectorized pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(src, indices, total_tokens, embed_dim):
    """src: [B, M, D], indices: [B, M] — scatter M tokens into [B, total_tokens, D] (zeros elsewhere)."""
    B, M, D = src.shape
    out = torch.zeros(B, total_tokens, D, dtype=src.dtype, device=src.device)
    out.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, D), src)
    return out


# --- KERNEL ---
@triton.jit
def _fused_token_scatter_fwd(
    SRC, IDX, OUT,
    M, D: tl.constexpr,
    total_tokens,
    BLOCK_D: tl.constexpr,
):
    # pid = b * M + m
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M

    # Load destination index
    idx = tl.load(IDX + b * M + m)

    src_base = SRC + (b * M + m) * D
    out_base = OUT + (b * total_tokens + idx) * D

    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        val = tl.load(src_base + cols, mask=mask, other=0.0)
        tl.store(out_base + cols, val, mask=mask)


def kernel_fn(src, indices, total_tokens, embed_dim):
    assert src.is_contiguous()
    assert indices.is_contiguous()
    B, M, D = src.shape
    BLOCK_D = min(triton.next_power_of_2(D), 4096)
    out = torch.zeros(B, total_tokens, D, dtype=src.dtype, device=src.device)
    total_programs = B * M
    _fused_token_scatter_fwd[(total_programs,)](
        src, indices, out,
        M=M, D=D,
        total_tokens=total_tokens,
        BLOCK_D=BLOCK_D,
        num_warps=min(16, max(1, BLOCK_D // 32)),
    )
    return out


def can_use_kernel(src, indices, total_tokens, embed_dim):
    return (
        src.is_cuda and indices.is_cuda
        and src.is_contiguous() and indices.is_contiguous()
        and src.ndim == 3
        and indices.ndim == 2
        and src.shape[:2] == indices.shape
        and src.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


SHAPES = {
    "small_mask":  {"src": (2, 196, 384),  "total_tokens": 784},
    "medium_mask": {"src": (2, 512, 1024), "total_tokens": 1568},
    "large_mask":  {"src": (2, 1024, 1280), "total_tokens": 3136},
}
