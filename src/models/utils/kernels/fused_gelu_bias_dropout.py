"""Fused GELU + Bias + Dropout kernel.

Pattern: y = dropout(GELU(x + bias), p)
Fuses: bias add + GELU activation + dropout mask into one kernel.
Common in FFN blocks during training where dropout follows GELU.
Uses pure scalar loads for generating per-element dropout masks.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, bias, p, training=True):
    y = torch.nn.functional.gelu(x + bias)
    if training:
        y = torch.nn.functional.dropout(y, p=p)
    return y


# --- KERNEL ---
@triton.jit
def _fused_gelu_bias_dropout_fwd(
    X, Bias, Y, Mask, DropoutScale,
    N: tl.constexpr, p: tl.constexpr,
    BLOCK: tl.constexpr,
    seed: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    b = tl.load(Bias + offs, mask=mask).to(tl.float32)
    y = x + b
    # GELU using sigmoid form: x * sigmoid(1.702 * x)
    k = 1.702
    sig = 1.0 / (1.0 + tl.exp(-k * y))
    gelu = y * sig
    # Dropout mask via philox random
    rand = tl.rand(seed, offs)
    keep = rand > p
    scale = 1.0 / (1.0 - p)
    y_drop = tl.where(keep, gelu, 0.0) * scale
    tl.store(Y + offs, y_drop, mask=mask)
    tl.store(Mask + offs, keep.to(tl.int8), mask=mask)
    if pid == 0:
        tl.store(DropoutScale, scale)


@triton.jit
def _fused_gelu_bias_dropout_bwd(
    X, Bias, DY, Mask, DX, DB,
    N: tl.constexpr, p: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    b = tl.load(Bias + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask).to(tl.float32)
    m = tl.load(Mask + offs, mask=mask).to(tl.int8)
    y = x + b
    # dGELU/dx
    k = 1.702
    sig = 1.0 / (1.0 + tl.exp(-k * y))
    dsig = sig * (1.0 - sig)
    dgelu = sig + k * y * dsig
    dx = dy * dgelu * m.to(tl.float32)
    db = dx
    tl.store(DX + offs, dx, mask=mask)
    tl.store(DB + offs, db, mask=mask)


class FusedGeluBiasDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, p=0.1, training=True):
        assert x.is_contiguous() and bias.is_contiguous()
        N = x.numel()
        y = torch.empty_like(x)
        mask = torch.empty(N, dtype=torch.int8, device=x.device)
        scale_tensor = torch.zeros(1, dtype=torch.float32, device=x.device)
        seed = torch.randint(2**31, [], device=x.device).item()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_bias_dropout_fwd[grid](
            x, bias, y, mask, scale_tensor, N, p, BLOCK=BLOCK, seed=seed,
            num_warps=4,
        )
        ctx.save_for_backward(x, bias, mask)
        ctx.p = p
        ctx.scale = scale_tensor.item()
        return y

    @staticmethod
    def backward(ctx, dy):
        x, bias, mask = ctx.saved_tensors
        N = x.numel()
        dx = torch.empty_like(x)
        db = torch.empty_like(bias)
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _fused_gelu_bias_dropout_bwd[grid](
            x, bias, dy, mask, dx, db, N, ctx.p, BLOCK=BLOCK,
            num_warps=4,
        )
        return dx, db, None, None


def kernel_fn(x, bias, p=0.1, training=True):
    return FusedGeluBiasDropout.apply(x, bias, p, training)


def can_use_kernel(x, bias, p=0.1):
    return (x.is_cuda and bias.is_cuda and
            x.is_contiguous() and bias.is_contiguous() and
            x.shape == bias.shape and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l": {"x": (2, 1024, 4096), "bias": (2, 1024, 4096)},
    "vit_h": {"x": (2, 2048, 5120), "bias": (2, 2048, 5120)},
    "small": {"x": (8, 256, 1536), "bias": (8, 256, 1536)},
}
