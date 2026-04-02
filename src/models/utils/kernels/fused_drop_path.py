"""Fused Drop Path kernel (stochastic depth).

Pattern: y = x if random > drop_prob else 0
Fuses: comparison + zeroing into one elementwise pass.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, drop_prob=0.1, training=True):
    return torch.nn.functional.dropout(x, p=drop_prob, training=training) if training else x


@triton.jit
def _drop_path_fwd(X, Y, Mask, N: tl.constexpr, DROP_PROB: tl.constexpr, BLOCK: tl.constexpr, seed: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.rand(seed, offs)
    keep = r > DROP_PROB
    scale = 1.0 / (1.0 - DROP_PROB)
    y = tl.where(keep, x * scale, 0.0)
    tl.store(Y + offs, y, mask=mask)
    tl.store(Mask + offs, keep.to(tl.int8), mask=mask)


@triton.jit
def _drop_path_bwd(X, Mask, DY, DX, N: tl.constexpr, DROP_PROB: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(Mask + offs, mask=mask, other=0.0).to(tl.int8)
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    scale = 1.0 / (1.0 - DROP_PROB)
    dx = tl.where(m, dy * scale, 0.0)
    tl.store(DX + offs, dx, mask=mask)


class FusedDropPath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, drop_prob=0.1, training=True):
        N = x.numel()
        y = torch.empty_like(x)
        mask = torch.empty(N, dtype=torch.int8, device=x.device)
        seed = torch.randint(2**31, [], device=x.device).item()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _drop_path_fwd[grid](x, y, mask, N, drop_prob, BLOCK=BLOCK, seed=seed, num_warps=4)
        ctx.save_for_backward(x, mask)
        ctx.drop_prob = drop_prob
        return y

    @staticmethod
    def backward(ctx, dy):
        x, mask = ctx.saved_tensors
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _drop_path_bwd[grid](x, mask, dy, dx, N, ctx.drop_prob, BLOCK=BLOCK, num_warps=4)
        return dx, None, None


def kernel_fn(x, drop_prob=0.1, training=True):
    return FusedDropPath.apply(x, drop_prob, training)


def can_use_kernel(x, drop_prob=0.1):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
