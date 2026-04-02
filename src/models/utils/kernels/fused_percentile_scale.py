"""Fused per-channel scale kernel.

Pattern: out = x * gamma where gamma is per-channel [C] and x is [B, N, C].
Fuses: channel-scale multiply without materializing gamma as same-shape tensor.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x, gamma):
    return x * gamma


# --- KERNEL ---
@triton.jit
def _fused_pct_scale_fwd(X, GAMMA, Y, B: tl.constexpr, N: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_b = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    for n in range(N):
        x = tl.load(X + pid_b * N * C + n * C + offs_c, mask=mask_c).to(tl.float32)
        g = tl.load(GAMMA + offs_c, mask=mask_c).to(tl.float32)
        y = x * g
        tl.store(Y + pid_b * N * C + n * C + offs_c, y, mask=mask_c)


@triton.jit
def _fused_pct_scale_bwd(X, GAMMA, DY, DX, B: tl.constexpr, N: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_b = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    for n in range(N):
        x = tl.load(X + pid_b * N * C + n * C + offs_c, mask=mask_c).to(tl.float32)
        g = tl.load(GAMMA + offs_c, mask=mask_c).to(tl.float32)
        dy = tl.load(DY + pid_b * N * C + n * C + offs_c, mask=mask_c).to(tl.float32)
        dx = dy * g
        tl.store(DX + pid_b * N * C + n * C + offs_c, dx, mask=mask_c)


class FusedPercentileScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma):
        assert x.is_contiguous() and gamma.is_contiguous()
        B, N, C = x.shape
        ctx.save_for_backward(x, gamma)
        y = torch.empty_like(x)
        BLOCK_C = triton.next_power_of_2(C)
        grid = (B,)
        _fused_pct_scale_fwd[grid](x, gamma, y, B, N, C, BLOCK_C=BLOCK_C, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, gamma = ctx.saved_tensors
        dy = dy.contiguous()
        B, N, C = x.shape
        dx = torch.empty_like(x)
        BLOCK_C = triton.next_power_of_2(C)
        grid = (B,)
        _fused_pct_scale_bwd[grid](x, gamma, dy, dx, B, N, C, BLOCK_C=BLOCK_C, num_warps=4)
        return dx, None


def kernel_fn(x, gamma):
    return FusedPercentileScale.apply(x, gamma)


def can_use_kernel(x, gamma):
    return (x.is_cuda and gamma.is_cuda and
            x.is_contiguous() and gamma.is_contiguous() and
            x.shape[-1] == gamma.shape[0] and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 1024), "gamma": (1024,)},
    "vit_h":  {"x": (2, 2048, 1280), "gamma": (1280,)},
    "small":  {"x": (8, 256, 384),   "gamma": (384,)},
}
