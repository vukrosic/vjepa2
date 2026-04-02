"""Fused LayerNorm split + concatenate kernel.

Source: app/vjepa_2_1/train.py:597-608
Pattern: hi_0 = LN(hi[..., :embed_dim]); hi_1 = LN(hi[..., embed_dim:2*embed_dim])
         hi_2 = LN(hi[..., 2*embed_dim:3*embed_dim]); hi_3 = LN(hi[..., -embed_dim:])
         hi_norm = cat([hi_0, hi_1, hi_2, hi_3], dim=-1)
Fuses: 4 LayerNorm calls + cat into a single kernel.
Instead of normalizing 4 times with separate kernel launches, we reshape to [B, N, 4, embed_dim]
and normalize all at once, then reshape back to [B, N, 4*embed_dim].
This is a 4x reduction in kernel launches for the normalization step.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, num_splits, weight, bias, eps=1e-5):
    """
    x: [B, N, num_splits * D], splits along last dim.
    weight/bias: [num_splits, D] or [D] (broadcast).
    Returns: concatenated normalized tensor [B, N, num_splits * D].
    """
    B, N, total_D = x.shape
    D = total_D // num_splits
    xs = x.view(B, N, num_splits, D)                    # [B, N, num_splits, D]
    xs = xs.transpose(1, 2).reshape(B * num_splits, N, D)  # [B, num_splits, N, D] -> [B*num_splits, N, D]
    normed = torch.nn.functional.layer_norm(xs, (D,), weight[:D] if weight.ndim > 1 else weight, bias[:D] if bias.ndim > 1 else bias, eps)
    normed = normed.view(B, num_splits, N, D).transpose(1, 2).reshape(B, N, total_D)
    return normed


# --- KERNEL ---
@triton.jit
def _ln_split_fwd_kernel(
    X, W, B, OUT,
    BATCH, N, num_splits, D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Grid: (BATCH * N * num_splits,)
    pid = ((b * N + n) * num_splits + s)
    """
    pid = tl.program_id(0)
    s = pid % num_splits
    tmp = pid // num_splits
    n = tmp % N
    b = tmp // N

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x_base = ((b * N + n) * num_splits + s) * D
    x = tl.load(X + x_base + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
    bb = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    out = (diff * inv_std) * w + bb

    out_base = ((b * N + n) * num_splits + s) * D
    tl.store(OUT + out_base + offs, out.to(x.dtype), mask=mask)


class FusedLayerNormSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, num_splits=4, eps=1e-5):
        B, N, total_D = x.shape
        D = total_D // num_splits
        BLOCK_D = triton.next_power_of_2(D)
        x_c = x.contiguous()
        out = torch.empty_like(x_c)
        n_programs = B * N * num_splits
        _ln_split_fwd_kernel[(n_programs,)](
            x_c, weight[:D].contiguous(), bias[:D].contiguous(), out,
            BATCH=B, N=N, num_splits=num_splits, D=D,
            BLOCK_D=BLOCK_D, eps=eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        ctx.save_for_backward(x_c, weight[:D], bias[:D])
        ctx.num_splits = num_splits; ctx.eps = eps
        ctx.B = B; ctx.N = N; ctx.D = D
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x_c, w, b = ctx.saved_tensors
        B, N, num_splits, D = ctx.B, ctx.N, ctx.num_splits, ctx.D
        eps = ctx.eps
        BLOCK_D = triton.next_power_of_2(D)
        dx = torch.empty_like(x_c)
        dw = torch.zeros(D, dtype=torch.float32, device=x_c.device)
        db = torch.zeros(D, dtype=torch.float32, device=x_c.device)

        @triton.jit
        def _bwd_kernel(X, W, DO, DX, DW, DB, BATCH, N, ns, D, BLOCK_D, eps):
            pid = tl.program_id(0)
            s = pid % ns
            tmp = pid // ns
            n = tmp % N
            b = tmp // N
            offs = tl.arange(0, BLOCK_D)
            mask = offs < D
            x = tl.load(X + ((b * N + n) * ns + s) * D + offs, mask=mask, other=0.0).to(tl.float32)
            mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
            diff = tl.where(mask, x - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / D
            inv_std = 1.0 / tl.sqrt(var + eps)
            x_hat = diff * inv_std
            w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
            do = tl.load(DO + ((b * N + n) * ns + s) * D + offs, mask=mask, other=0.0).to(tl.float32)
            dx_norm = do * w
            dvar = tl.sum(dx_norm * diff * (-0.5) * inv_std**3, axis=0)
            dmean = tl.sum(-dx_norm * inv_std, axis=0) + dvar * tl.sum(-2.0 * diff, axis=0) / D
            dxi = dx_norm * inv_std + dvar * 2.0 * diff / D + dmean / D
            dxi = tl.where(mask, dxi, 0.0)
            tl.store(DX + ((b * N + n) * ns + s) * D + offs, dxi.to(x.dtype), mask=mask)
            tl.atomic_add(DW + offs, tl.where(mask, do * x_hat, 0.0))
            tl.atomic_add(DB + offs, tl.where(mask, do, 0.0))

        _bwd_kernel[(B * N * num_splits,)](
            x_c, w, grad_out, dx, dw, db,
            B, N, num_splits, D, BLOCK_D=BLOCK_D, eps=eps,
            num_warps=min(16, max(1, BLOCK_D // 32)),
        )
        return dx, dw.to(w.dtype), db.to(b.dtype), None, None


def kernel_fn(x, weight, bias, num_splits=4, eps=1e-5):
    return FusedLayerNormSplit.apply(x, weight, bias, num_splits, eps)


def can_use_kernel(x, weight, bias, num_splits=4, eps=1e-5):
    if not x.is_cuda:
        return False
    if x.shape[-1] % num_splits != 0:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


SHAPES = {
    "vit_l_4way": {"x": (2, 784, 4096), "num_splits": 4, "D": 1024},
    "vit_h_4way": {"x": (2, 1024, 5120), "num_splits": 4, "D": 1280},
    "small_4way": {"x": (2, 196, 1536), "num_splits": 4, "D": 384},
}
