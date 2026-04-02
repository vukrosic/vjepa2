"""Fused BatchNorm kernel.

Pattern: y = (x - mean) / sqrt(var + eps) * weight + bias
Pure scalar loads per channel.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, weight, bias, running_mean, running_var, eps=1e-5, momentum=0.1, training=False):
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=training, momentum=momentum, eps=eps)


@triton.jit
def _batch_norm_fwd(X, Y, W, B, RM, RV, N: tl.constexpr, C: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(0)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + c, mask=None).to(tl.float32)
    b = tl.load(B + c, mask=None).to(tl.float32)
    rm = tl.load(RM + c, mask=None).to(tl.float32)
    rv = tl.load(RV + c, mask=None).to(tl.float32)
    y = (x - rm) / tl.sqrt(rv + EPS) * w + b
    tl.store(Y + offs, y, mask=mask)


class FusedBatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, eps=1e-5, momentum=0.1, training=False):
        C = x.shape[1]
        N = x.numel() // C
        y = torch.empty_like(x)
        BLOCK = 1024
        n_blocks = (N + BLOCK - 1) // BLOCK
        _batch_norm_fwd[(C,)](x, y, weight, bias, running_mean, running_var, N, C, eps, BLOCK=BLOCK, num_warps=4)
        return y


def kernel_fn(x, weight, bias, running_mean, running_var, eps=1e-5, momentum=0.1, training=False):
    return FusedBatchNorm.apply(x, weight, bias, running_mean, running_var, eps, momentum, training)


def can_use_kernel(x, weight, bias, running_mean, running_var):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 4096, 64, 64), "C": 4096},
    "vit_h":  {"x": (2, 5120, 32, 32), "C": 5120},
    "small":  {"x": (8, 1536, 16, 16), "C": 1536},
}
