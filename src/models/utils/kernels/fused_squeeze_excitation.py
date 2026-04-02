"""Fused Squeeze-and-Excitation elementwise kernel.

Pattern: y = x * sigmoid(se) where se is precomputed SE excitation.
Fuses: sigmoid + multiply into one elementwise pass.
Note: matmul (fc1, fc2) is done via cuBLAS in PyTorch; this kernel handles the elementwise part.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x, se):
    """x: [B, N, C], se: [B, 1, C]"""
    return x * torch.sigmoid(se)


# --- KERNEL ---
@triton.jit
def _fused_se_fwd(X, SE, Y, B: tl.constexpr, N: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_b = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    for n in range(N):
        # Block-level pointer arithmetic
        x_base = pid_b * N * C + n * C
        x_ptrs = x_base + offs_c
        se_base = pid_b * C
        se_ptrs = se_base + offs_c
        x = tl.load(X + x_ptrs, mask=mask_c).to(tl.float32)
        s = tl.load(SE + se_ptrs, mask=mask_c).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-s))
        y = x * sig
        tl.store(Y + x_ptrs, y, mask=mask_c)


@triton.jit
def _fused_se_bwd(X, SE, DY, DX, B: tl.constexpr, N: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_b = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    for n in range(N):
        # Block-level pointer arithmetic
        x_base = pid_b * N * C + n * C
        x_ptrs = x_base + offs_c
        se_base = pid_b * C
        se_ptrs = se_base + offs_c
        dy_base = pid_b * N * C + n * C
        dy_ptrs = dy_base + offs_c
        x = tl.load(X + x_ptrs, mask=mask_c).to(tl.float32)
        s = tl.load(SE + se_ptrs, mask=mask_c).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-s))
        dy = tl.load(DY + dy_ptrs, mask=mask_c).to(tl.float32)
        # d(x*sig)/dx = sig + x * sig * (1 - sig) = sig * (1 + x * (1 - sig))
        dsig_ds = sig * (1.0 - sig)
        dx = dy * sig + dy * x * dsig_ds
        # d(x*sig)/dse = x * sig * (1 - sig)
        tl.store(DX + x_ptrs, dx, mask=mask_c)


class FusedSqueezeExcitation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, se):
        assert x.is_contiguous() and se.is_contiguous()
        B, N, C = x.shape
        ctx.save_for_backward(x, se)
        y = torch.empty_like(x)
        BLOCK_C = triton.next_power_of_2(C)
        grid = (B,)
        _fused_se_fwd[grid](x, se, y, B, N, C, BLOCK_C=BLOCK_C, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, se = ctx.saved_tensors
        dy = dy.contiguous()
        B, N, C = x.shape
        dx = torch.empty_like(x)
        BLOCK_C = triton.next_power_of_2(C)
        grid = (B,)
        _fused_se_bwd[grid](x, se, dy, dx, B, N, C, BLOCK_C=BLOCK_C, num_warps=4)
        return dx, None


def kernel_fn(x, se):
    return FusedSqueezeExcitation.apply(x, se)


def can_use_kernel(x, se):
    return (x.is_cuda and se.is_cuda and
            x.is_contiguous() and se.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            se.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 1024), "se": (2, 1, 1024)},
    "vit_h":  {"x": (2, 2048, 1280), "se": (2, 1, 1280)},
    "small":  {"x": (8, 256, 384),   "se": (8, 1, 384)},
}
