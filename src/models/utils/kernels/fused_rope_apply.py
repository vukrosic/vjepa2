"""Fused RoPE (Rotary Position Embedding) apply kernel.

Applies RoPE rotation to a single tensor x: [B, H, N, D].
Each (B, H, N) triple is handled by one Triton program; D features are
processed in a single vectorized pass with inline frequency computation.

Compared to the existing multi-kernel approach, this standalone fused kernel
avoids intermediate tensor allocation for cos/sin tables at the call site
and is useful for benchmarking the rotation primitive itself.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, pos, head_dim):
    """
    x: [B, H, N, D], pos: [N] positions, head_dim: int
    Standard RoPE rotation using the "cat" approach.
    """
    B, H, N, D = x.shape
    half = D // 2
    omega = 1.0 / (10000 ** (torch.arange(half, dtype=x.dtype, device=x.device) / half))
    freq = pos.to(x.dtype).unsqueeze(-1) * omega  # [N, half]
    cos = freq.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, N, half]
    sin = freq.sin().unsqueeze(0).unsqueeze(0)
    cos = torch.cat([cos, cos], dim=-1)  # [1, 1, N, D]
    sin = torch.cat([sin, sin], dim=-1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# --- KERNEL ---
@triton.jit
def _fused_rope_apply_fwd(
    X, POS, OUT,
    stride_b, stride_h, stride_n,
    N, D: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    # pid maps to (b, h, n)
    pid = tl.program_id(0)
    # Decompose pid -> b, h, n
    # stride layout: X is [B, H, N, D] contiguous
    # pid = b*(H*N) + h*N + n  — but we just use the linear row index
    # since stride_n already accounts for H and B offsets when we treat
    # each (b,h,n) as an independent "row" of length D.

    # Load position for this (b, h, n) triple.
    # All (b, h) sharing the same n have the same position.
    # n = pid % N
    n = pid % N
    p = tl.load(POS + n).to(tl.float32)

    base = pid * D  # linear offset into X / OUT

    # Process each frequency band in BLOCK_HALF chunks
    for off in range(0, HALF, BLOCK_HALF):
        freq_idx = off + tl.arange(0, BLOCK_HALF)  # [0..BLOCK_HALF)
        mask = freq_idx < HALF

        # omega = 1 / 10000^(freq_idx / HALF)
        omega = tl.exp(-tl.log(10000.0) * freq_idx.to(tl.float32) / HALF)
        theta = p * omega  # [BLOCK_HALF]
        c = tl.cos(theta)
        s = tl.sin(theta)

        # Load x1 (first half), x2 (second half)
        x1 = tl.load(X + base + freq_idx, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X + base + HALF + freq_idx, mask=mask, other=0.0).to(tl.float32)

        # RoPE: out_1 = x1*cos - x2*sin, out_2 = x2*cos + x1*sin
        out1 = x1 * c - x2 * s
        out2 = x2 * c + x1 * s

        tl.store(OUT + base + freq_idx, out1, mask=mask)
        tl.store(OUT + base + HALF + freq_idx, out2, mask=mask)


def kernel_fn(x, pos, head_dim):
    assert x.is_contiguous()
    assert pos.is_contiguous()
    B, H, N, D = x.shape
    half = D // 2
    BLOCK_HALF = min(triton.next_power_of_2(half), 2048)
    out = torch.empty_like(x)
    total_programs = B * H * N
    _fused_rope_apply_fwd[(total_programs,)](
        x, pos, out,
        x.stride(0), x.stride(1), x.stride(2),
        N=N, D=D, HALF=half, BLOCK_HALF=BLOCK_HALF,
        num_warps=min(16, max(1, BLOCK_HALF // 16)),
    )
    return out


def can_use_kernel(x, pos, head_dim):
    return (
        x.is_cuda and pos.is_cuda
        and x.is_contiguous() and pos.is_contiguous()
        and x.ndim == 4
        and x.shape[-1] % 2 == 0
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


SHAPES = {
    "vit_l_d": {"x": (2, 16, 1024, 64), "pos": (1024,), "head_dim": 64},
    "vit_h_d": {"x": (2, 16, 2048, 80), "pos": (2048,), "head_dim": 80},
    "small":   {"x": (4, 12, 256, 64),  "pos": (256,),  "head_dim": 64},
}
