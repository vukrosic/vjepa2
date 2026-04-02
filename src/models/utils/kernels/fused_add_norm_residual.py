"""Fused add + LayerNorm for two-branch residual blocks.

Pattern: out = LayerNorm(x + branch1 + branch2)
Fuses: 3-input add + LayerNorm in one pass per row.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(x, b1, b2, weight, bias, eps=1e-5):
    y = x + b1 + b2
    return torch.nn.functional.layer_norm(y, (y.shape[-1],), weight, bias, eps)


# --- KERNEL ---
@triton.jit
def _fused_add_norm_fwd(
    X, B1, B2, W, B_out, Y,
    BN: tl.constexpr, D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Grid: (BN,) — one program per row of length D.

    Loads full row in one block, computes mean/variance,
    then normalizes and stores.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    # row_base as block-level tensor to match offs arithmetic
    row_base = offs + pid * D

    x0 = tl.load(X + row_base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(B1 + row_base, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(B2 + row_base, mask=mask, other=0.0).to(tl.float32)
    row = x0 + x1 + x2

    mean = tl.sum(row, axis=0) / D
    var = tl.sum((row - mean) * (row - mean), axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + EPS)

    norm = (row - mean) * inv_std
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_out + offs, mask=mask, other=0.0).to(tl.float32)
    out = norm * w + b
    tl.store(Y + row_base, out, mask=mask)


class FusedAddNormResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b1, b2, weight, bias, eps=1e-5):
        assert x.is_contiguous() and b1.is_contiguous() and b2.is_contiguous()
        orig_shape = x.shape
        x_flat = x.reshape(-1)
        b1_flat = b1.reshape(-1)
        b2_flat = b2.reshape(-1)
        N_total = x_flat.numel()
        BN = N_total // x.shape[-1]
        D = x.shape[-1]
        ctx.save_for_backward(x_flat, b1_flat, b2_flat, weight, bias)
        ctx.eps = eps
        ctx.BN = BN
        ctx.D = D
        y_flat = torch.empty_like(x_flat)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (BN,)
        _fused_add_norm_fwd[grid](
            x_flat, b1_flat, b2_flat, weight, bias, y_flat,
            BN=BN, D=D, EPS=eps, BLOCK_D=BLOCK_D, num_warps=4,
        )
        return y_flat.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        # Use PyTorch for backward (fused kernel is for forward speed)
        orig_shape = dy.shape
        x, b1, b2, weight, bias = ctx.saved_tensors
        BN, D = ctx.BN, ctx.D
        del BN
        B, N, D = orig_shape[0], orig_shape[1], orig_shape[2]
        N = orig_shape.numel() // (B * D)
        eps = ctx.eps

        # Reshape to (B, N, D) for proper LayerNorm backward
        row = x.reshape(B, N, D) + b1.reshape(B, N, D) + b2.reshape(B, N, D)
        mean = row.sum(dim=-1, keepdim=True) / D
        var = ((row - mean) ** 2).sum(dim=-1, keepdim=True) / D
        inv_std = 1.0 / torch.sqrt(var + eps)
        norm = (row - mean) * inv_std

        dy_3d = dy.reshape(B, N, D)
        dnorm = dy_3d * weight  # broadcast weight (D,) over (B, N, D)
        dvar = (dnorm * (row - mean) * (-0.5) * inv_std.pow(3)).sum(dim=-1, keepdim=True)
        dmean = (dnorm * (-inv_std)).sum(dim=-1, keepdim=True) + \
                dvar * (-2.0 * (row - mean)).sum(dim=-1, keepdim=True) / D
        drow = dnorm * inv_std + dvar * 2.0 * (row - mean) / D + dmean / D

        dx = drow.clone()
        db1 = drow.clone()
        db2 = drow.clone()
        dw = (dy_3d * norm).sum(dim=(0, 1))  # (D,)
        db = dy_3d.sum(dim=(0, 1))

        return dx.reshape(orig_shape), db1.reshape(orig_shape), db2.reshape(orig_shape), dw, db, None


def kernel_fn(x, b1, b2, weight, bias, eps=1e-5):
    return FusedAddNormResidual.apply(x, b1, b2, weight, bias, eps)


def can_use_kernel(x, b1, b2, weight, bias):
    return (x.is_cuda and b1.is_cuda and b2.is_cuda and weight.is_cuda and bias.is_cuda and
            x.is_contiguous() and b1.is_contiguous() and b2.is_contiguous() and
            x.shape == b1.shape == b2.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 1024), "b1": (2, 1024, 1024), "b2": (2, 1024, 1024), "D": 1024},
    "vit_h":  {"x": (2, 2048, 1280), "b1": (2, 2048, 1280), "b2": (2, 2048, 1280), "D": 1280},
    "small":  {"x": (8, 256, 384),   "b1": (8, 256, 384),   "b2": (8, 256, 384),   "D": 384},
}
