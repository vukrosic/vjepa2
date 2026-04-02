"""Fused batch_norm_1d kernel.

Pattern: torch.nn.functional.batch_norm(x, running_mean=None, running_var=None, training=True)
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x):
    return torch.nn.functional.batch_norm(x, running_mean=None, running_var=None, training=True)


@triton.jit
def _fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_base = row * N
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    mean = 0.0
    m2 = 0.0
    for c in range(N):
        vc = tl.load(X + row_base + c, mask=c < N, other=0.0).to(tl.float32)
        mean = mean + vc
        m2 = m2 + vc * vc
    mean = mean / N
    var = m2 / N - mean * mean
    denom = 1.0 / (tl.sqrt(var + 1e-5))
    v_self = tl.load(X + row_base + offs, mask=mask, other=0.0).to(tl.float32)
    y = (v_self - mean) * denom
    tl.store(Y + row_base + offs, y, mask=mask)


@triton.jit
def _bwd(X, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_base = row * N
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    mean = 0.0
    m2 = 0.0
    for c in range(N):
        vc = tl.load(X + row_base + c, mask=c < N, other=0.0).to(tl.float32)
        mean = mean + vc
        m2 = m2 + vc * vc
    mean = mean / N
    var = m2 / N - mean * mean
    denom = 1.0 / (tl.sqrt(var + 1e-5))
    v_self = tl.load(X + row_base + offs, mask=mask, other=0.0).to(tl.float32)
    x_norm = (v_self - mean) * denom
    dnorm = tl.load(DY + row_base + offs, mask=mask, other=0.0).to(tl.float32)
    dmean = 0.0
    for c in range(N):
        dvc = tl.load(DY + row_base + c, mask=c < N, other=0.0).to(tl.float32)
        dmean = dmean + dvc
    dx = dnorm * denom - (x_norm * dmean) / N - dmean * mean / N
    tl.store(DX + row_base + offs, dx, mask=mask)


class BatchNorm1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        BLOCK = 1024
        grid = (x.shape[0],)
        _fwd[grid](x, y, x.shape[-1], BLOCK=BLOCK, num_warps=4)

        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        grid = (x.shape[0],)
        _bwd[grid](x, dy, dx, x.shape[-1], BLOCK=BLOCK, num_warps=4)
        return dx


def kernel_fn(x):
    return BatchNorm1D.apply(x)


def can_use_kernel(x):
    return (x.is_cuda and x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 1024, 4096)},
    "vit_h":  {"x": (2, 2048, 5120)},
    "small":  {"x": (8, 256, 1536)},
}
