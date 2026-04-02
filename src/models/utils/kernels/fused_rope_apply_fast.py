"""Fused fast RoPE apply kernel using vectorized sin/cos.

Pattern: RoPE rotation applied to x with precomputed cos/sin.
Fuses: vectorized (2-element) sin/cos compute + rotation in one pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x, pos, head_dim):
    B, H, N, D = x.shape
    half = D // 2
    omega = 1.0 / (10000 ** (torch.arange(half, dtype=x.dtype, device=x.device) / half))
    freq = pos.float().unsqueeze(-1) * omega
    cos = freq.cos().unsqueeze(0).unsqueeze(0).expand(B, H, N, half)
    sin = freq.sin().unsqueeze(0).unsqueeze(0).expand(B, H, N, half)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# --- KERNEL ---
@triton.jit
def _fused_rope_fwd(X, COS, SIN, Y, B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_d = tl.arange(0, BLOCK_D)
    half = D // 2
    mask_d = offs_d < D
    for d0 in range(0, half, BLOCK_D // 2):
        # Even pair: (d0, d0+half)
        d_e = d0 + offs_d
        d_o = d0 + half + offs_d
        mask_e = d_e < half
        mask_o = d_o < D
        # Load x1, x2
        x1 = tl.load(X + pid_b * H * N * D + pid_h * N * D + pid_n * D + d_e, mask=mask_e, other=0.0).to(tl.float32)
        x2 = tl.load(X + pid_b * H * N * D + pid_h * N * D + pid_n * D + d_o, mask=mask_o, other=0.0).to(tl.float32)
        # Load cos, sin (stored compact at [N, half])
        cos_e = tl.load(COS + pid_n * half + d_e, mask=mask_e, other=0.0).to(tl.float32)
        sin_e = tl.load(SIN + pid_n * half + d_e, mask=mask_e, other=0.0).to(tl.float32)
        # Apply rotation: x1' = x1 * cos - x2 * sin, x2' = x1 * sin + x2 * cos
        x1r = x1 * cos_e - x2 * sin_e
        x2r = x1 * sin_e + x2 * cos_e
        tl.store(Y + pid_b * H * N * D + pid_h * N * D + pid_n * D + d_e, x1r, mask=mask_e)
        tl.store(Y + pid_b * H * N * D + pid_h * N * D + pid_n * D + d_o, x2r, mask=mask_o)


def _precompute_cos_sin(pos, half, device, dtype):
    omega = 1.0 / (10000 ** (torch.arange(half, dtype=dtype, device=device) / half))
    freq = pos.float().unsqueeze(-1) * omega
    return freq.cos(), freq.sin()


def kernel_fn(x, pos, head_dim):
    B, H, N, D = x.shape
    half = D // 2
    cos, sin = _precompute_cos_sin(pos, half, x.device, x.dtype)
    y = torch.empty_like(x)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B, H, N)
    _fused_rope_fwd[grid](x, cos, sin, y, B, H, N, D, BLOCK_D=BLOCK_D, num_warps=4)
    return y


def can_use_kernel(x, pos, head_dim):
    return (x.is_cuda and pos.is_cuda and
            x.is_contiguous() and pos.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            head_dim <= 128)


SHAPES = {
    "vit_l64":  {"x": (2, 16, 1024, 64),  "pos": (1024,), "head_dim": 64},
    "vit_h80":  {"x": (2, 16, 2048, 80),  "pos": (2048,), "head_dim": 80},
    "small64":  {"x": (4, 12, 256, 64),   "pos": (256,),  "head_dim": 64},
}
