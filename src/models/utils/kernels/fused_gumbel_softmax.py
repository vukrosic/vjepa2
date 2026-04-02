"""Fused Gumbel Softmax kernel.

Pattern: Gumbel-Softmax samples from Gumbel(0,1) - log(-log(Uniform(0,1)))
Fuses: uniform + log + neg + add + softmax into one pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, tau=1.0, hard=False, dim=-1):
    return torch.nn.functional.gumbel_softmax(x, tau=tau, hard=hard, dim=dim)


# --- KERNEL ---
@triton.jit
def _gumbel_softmax_fwd(X, U, Y, N: tl.constexpr, TAU: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    u = tl.load(U + offs, mask=mask).to(tl.float32)
    gumbel = -tl.log(tl.minimum(-tl.log(tl.maximum(u, 1e-10)), -1e-10) + 1e-10)
    logits = x + gumbel
    # Numerically stable softmax
    max_logits = tl.max(logits)
    e = tl.exp(logits - max_logits)
    sum_e = tl.sum(e)
    y = e / sum_e
    tl.store(Y + offs, y, mask=mask)


class FusedGumbelSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau=1.0, hard=False, dim=-1):
        assert x.is_contiguous()
        assert dim == -1
        ctx.save_for_backward(x)
        ctx.tau = tau
        ctx.hard = hard
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        # For softmax we need to be per-row, so use rows as grid
        rows = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1]
        grid = (rows,)
        # Uniform samples - in practice would need different random source
        u = torch.rand_like(x)
        _gumbel_softmax_fwd[grid](x, u, y, N, tau, BLOCK=BLOCK, num_warps=4)
        if hard:
            # Straight-through gradient: set max to 1, others to 0
            index = y.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(dim=-1, index=1, value=1.0)
            return y_hard - y.detach() + y
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("GumbelSoftmax backward not yet implemented")


def kernel_fn(x, tau=1.0, hard=False, dim=-1):
    return FusedGumbelSoftmax.apply(x, tau, hard, dim)


def can_use_kernel(x, tau=1.0, hard=False, dim=-1):
    return (x.is_cuda and
            x.is_contiguous() and
            dim == -1 and
            x.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
