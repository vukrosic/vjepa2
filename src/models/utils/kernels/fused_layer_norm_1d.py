"""Fused LayerNorm-1D kernel.

Pattern: y = (x - mean) / sqrt(var + eps) * weight + bias
Normalizes over last dimension.
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, weight, bias, eps=1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)


# --- KERNEL ---
@triton.jit
def _layer_norm_fwd(X, W, B, Y, N: tl.constexpr, LAST_DIM: tl.constexpr,
                    EPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        row_off = offs * LAST_DIM
        # Compute mean
        s = 0.0
        for j in range(LAST_DIM):
            v = tl.load(X + row_off + j).to(tl.float32)
            s = s + v
        mean = s / tl.cast(LAST_DIM, tl.float32)
        # Compute variance
        var_s = 0.0
        for j in range(LAST_DIM):
            v = tl.load(X + row_off + j).to(tl.float32)
            d = v - mean
            var_s = var_s + d * d
        inv_std = 1.0 / tl.sqrt(var_s / tl.cast(LAST_DIM, tl.float32) + EPS)
        # Normalize
        for j in range(LAST_DIM):
            v = tl.load(X + row_off + j).to(tl.float32)
            w = tl.load(W + j).to(tl.float32) if W != 0 else 1.0
            b = tl.load(B + j).to(tl.float32) if B != 0 else 0.0
            y = (v - mean) * inv_std * w + b
            tl.store(Y + row_off + j, y)


@triton.jit
def _layer_norm_bwd(X, W, DY, DX, N: tl.constexpr, LAST_DIM: tl.constexpr,
                    EPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        row_off = offs * LAST_DIM
        # Recompute mean and inv_std
        s = 0.0
        for j in range(LAST_DIM):
            v = tl.load(X + row_off + j).to(tl.float32)
            s = s + v
        mean = s / tl.cast(LAST_DIM, tl.float32)
        var_s = 0.0
        for j in range(LAST_DIM):
            v = tl.load(X + row_off + j).to(tl.float32)
            d = v - mean
            var_s = var_s + d * d
        inv_std = 1.0 / tl.sqrt(var_s / tl.cast(LAST_DIM, tl.float32) + EPS)
        # Weighted gradient sum
        w_sum = 0.0
        for j in range(LAST_DIM):
            w = tl.load(W + j).to(tl.float32) if W != 0 else 1.0
            dy = tl.load(DY + row_off + j).to(tl.float32)
            w_sum = w_sum + dy * w
        scale = w_sum * inv_std / tl.cast(LAST_DIM, tl.float32)
        # Compute dx
        for j in range(LAST_DIM):
            dy = tl.load(DY + row_off + j).to(tl.float32)
            w = tl.load(W + j).to(tl.float32) if W != 0 else 1.0
            dx = (dy * w - scale) * inv_std
            tl.store(DX + row_off + j, dx)


class FusedLayerNorm1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        assert x.is_contiguous()
        ctx.save_for_backward(x, weight, bias)
        ctx.eps = eps
        y = torch.empty_like(x)
        N = x.shape[0]
        LAST_DIM = x.shape[-1]
        BLOCK = 256
        grid = ((N + BLOCK - 1) // BLOCK,)
        _layer_norm_fwd[grid](x, weight, bias, y, N, LAST_DIM, eps, BLOCK=BLOCK, num_warps=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.shape[0]
        LAST_DIM = x.shape[-1]
        BLOCK = 256
        grid = ((N + BLOCK - 1) // BLOCK,)
        _layer_norm_bwd[grid](x, weight, dy, dx, N, LAST_DIM, ctx.eps, BLOCK=BLOCK, num_warps=4)
        return dx, None, None


def kernel_fn(x, weight, bias, eps=1e-5):
    return FusedLayerNorm1D.apply(x, weight, bias, eps)


def can_use_kernel(x, weight, bias, eps=1e-5):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16) and x.dim() == 2)


SHAPES = {
    "vit_l":  {"x": (1024, 4096)},
    "vit_h":  {"x": (2048, 5120)},
    "small":  {"x": (256, 1536)},
}
