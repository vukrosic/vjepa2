"""Fused LayerScale + residual add kernel.

LayerScale multiplies sub-layer output by a learnable per-channel scalar gamma,
then adds to the residual stream. Fusing both ops avoids one full read/write
pass over the data vs. doing them separately.

  out = residual + x * gamma

Used in DeiT-style and some V-JEPA variants with layer-scale stabilization.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, gamma, residual):
    return residual + x * gamma


# --- FORWARD KERNEL ---
@triton.jit
def _fused_layer_scale_fwd(
    X, GAMMA, RESIDUAL, OUT,
    N_ROWS, D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per row
    row_off = pid * D

    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(GAMMA + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(RESIDUAL + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        out = r + x * g
        tl.store(OUT + row_off + cols, out, mask=mask)


# --- BACKWARD KERNEL ---
@triton.jit
def _fused_layer_scale_bwd(
    X, GAMMA, DY, DX, DGAMMA, DRESIDUAL,
    N_ROWS, D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per row
    row_off = pid * D

    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(GAMMA + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row_off + cols, mask=mask, other=0.0).to(tl.float32)

        # dx = dy * gamma
        dx = dy * g
        tl.store(DX + row_off + cols, dx, mask=mask)

        # dresidual = dy (identity pass-through)
        tl.store(DRESIDUAL + row_off + cols, dy, mask=mask)

        # dgamma accumulation: atomic add per row
        dgamma = dy * x
        tl.atomic_add(DGAMMA + cols, dgamma, mask=mask)


class FusedLayerScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, residual):
        assert x.is_contiguous()
        assert gamma.is_contiguous()
        assert residual.is_contiguous()
        orig_shape = x.shape
        D = x.shape[-1]
        x2d = x.view(-1, D)
        res2d = residual.view(-1, D)
        N_ROWS = x2d.shape[0]
        BLOCK_D = min(triton.next_power_of_2(D), 4096)
        out2d = torch.empty_like(x2d)
        _fused_layer_scale_fwd[(N_ROWS,)](
            x2d, gamma, res2d, out2d,
            N_ROWS=N_ROWS, D=D, BLOCK_D=BLOCK_D,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        ctx.save_for_backward(x2d, gamma)
        ctx.D = D
        ctx.N_ROWS = N_ROWS
        ctx.BLOCK_D = BLOCK_D
        ctx.orig_shape = orig_shape
        return out2d.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, gamma = ctx.saved_tensors
        D, N_ROWS, BLOCK_D = ctx.D, ctx.N_ROWS, ctx.BLOCK_D
        dy2d = dy.contiguous().view(N_ROWS, D)
        dx2d = torch.empty_like(x2d)
        dgamma = torch.zeros(D, dtype=torch.float32, device=x2d.device)
        dresidual2d = torch.empty_like(x2d)
        _fused_layer_scale_bwd[(N_ROWS,)](
            x2d, gamma, dy2d, dx2d, dgamma, dresidual2d,
            N_ROWS=N_ROWS, D=D, BLOCK_D=BLOCK_D,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        return (
            dx2d.view(ctx.orig_shape),
            dgamma.to(gamma.dtype),
            dresidual2d.view(ctx.orig_shape),
        )


def kernel_fn(x, gamma, residual):
    return FusedLayerScale.apply(x, gamma, residual)


def can_use_kernel(x, gamma, residual):
    return (
        x.is_cuda and gamma.is_cuda and residual.is_cuda
        and x.is_contiguous() and gamma.is_contiguous() and residual.is_contiguous()
        and x.shape == residual.shape
        and x.shape[-1] == gamma.shape[0]
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


SHAPES = {
    "vit_l": {"x": (2, 1024, 1024), "gamma": (1024,), "residual": (2, 1024, 1024)},
    "vit_h": {"x": (2, 2048, 1280), "gamma": (1280,), "residual": (2, 2048, 1280)},
    "small": {"x": (8, 256, 384),   "gamma": (384,),   "residual": (8, 256, 384)},
}
