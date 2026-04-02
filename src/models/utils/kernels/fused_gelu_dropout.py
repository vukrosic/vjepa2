"""Fused GELU + Dropout kernel.

Source: src/models/utils/modules.py:153-158 (MLP.forward)
Pattern: x = gelu(fc1(x)); x = dropout(x); x = fc2(x)
Fuses: GELU activation + Bernoulli dropout mask into one kernel.
Saves one full read/write pass over the activation tensor.
Frequency: 1x per non-SwiGLU block.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from source) ---
def baseline_fn(x, drop_p, training):
    x = torch.nn.functional.gelu(x)
    x = torch.nn.functional.dropout(x, p=drop_p, training=training)
    return x


# --- KERNEL ---
@triton.jit
def _fused_gelu_dropout_fwd(
    X, Y, MASK,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    drop_p: tl.constexpr,
    seed,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU approx (tanh form)
    c = 0.7978845608028654  # sqrt(2/pi)
    y = 0.5 * x * (1.0 + tl.math.tanh(c * (x + 0.044715 * x * x * x)))

    # Dropout using Philox RNG
    keep = tl.rand(seed, offs) > drop_p
    scale = 1.0 / (1.0 - drop_p)
    y = tl.where(keep, y * scale, 0.0)

    tl.store(Y + offs, y.to(x.dtype), mask=mask)
    tl.store(MASK + offs, keep.to(tl.int8), mask=mask)


@triton.jit
def _fused_gelu_dropout_bwd(
    X, DY, DX, MASK,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    drop_p: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(MASK + offs, mask=mask, other=0).to(tl.int1)

    # GELU grad
    c = 0.7978845608028654
    t = tl.math.tanh(c * (x + 0.044715 * x * x * x))
    dtanh = 1.0 - t * t
    dgelu = 0.5 * (1.0 + t) + 0.5 * x * dtanh * c * (1.0 + 3.0 * 0.044715 * x * x)

    # Dropout mask scale
    scale = 1.0 / (1.0 - drop_p)
    dx = dy * dgelu * tl.where(keep, scale, 0.0)

    tl.store(DX + offs, dx.to(x.dtype), mask=mask)


class FusedGELUDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, drop_p, training):
        if not training or drop_p == 0.0:
            y = torch.nn.functional.gelu(x)
            ctx.save_for_backward(x)
            ctx.drop_p = 0.0
            ctx.mask = None
            return y

        x_c = x.contiguous()
        y = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        drop_mask = torch.empty(N, dtype=torch.int8, device=x.device)

        seed = torch.randint(0, 2**31, (1,), device="cpu").item()
        _fused_gelu_dropout_fwd[(n_blocks,)](
            x_c, y, drop_mask, N=N, BLOCK=BLOCK, drop_p=drop_p, seed=seed,
            num_warps=4,
        )
        ctx.save_for_backward(x_c, drop_mask)
        ctx.drop_p = drop_p
        return y

    @staticmethod
    def backward(ctx, dy):
        if ctx.drop_p == 0.0:
            (x_c,) = ctx.saved_tensors
            # GELU backward only
            t = torch.tanh(0.7978845608 * (x_c + 0.044715 * x_c ** 3))
            dgelu = 0.5 * (1.0 + t) + 0.5 * x_c * (1.0 - t * t) * 0.7978845608 * (1.0 + 3.0 * 0.044715 * x_c ** 2)
            return dy * dgelu, None, None

        x_c, drop_mask = ctx.saved_tensors
        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        N = x_c.numel()
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK

        _fused_gelu_dropout_bwd[(n_blocks,)](
            x_c, dy_c, dx, drop_mask, N=N, BLOCK=BLOCK, drop_p=ctx.drop_p,
            num_warps=4,
        )
        return dx, None, None


def kernel_fn(x, drop_p=0.0, training=False):
    return FusedGELUDropout.apply(x, drop_p, training)


def can_use_kernel(x, drop_p=0.0, training=False):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l_mlp":  {"x": (2, 1024, 4096)},   # ViT-L hidden (4x MLP expansion)
    "vit_h_mlp":  {"x": (2, 2048, 5120)},   # ViT-H hidden
    "small":      {"x": (8, 256, 1536)},     # small batch
}
