"""Fused InstanceNorm kernel.

Pattern: y = (x - mean) / sqrt(var + eps) * weight + bias
Instance norm applied per sample per channel.
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(x, weight, bias, eps=1e-5):
    return torch.nn.functional.instance_norm(x, weight=weight, bias=bias, eps=eps)


@triton.jit
def _instance_norm_fwd(X, Y, W, B, N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W_: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    c = pid
    offs = tl.arange(0, BLOCK)
    mask = offs < N * H * W_
    x = tl.load(X + c * N * H * W_ + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x) / (N * H * W_)
    var = tl.sum((x - mean) * (x - mean)) / (N * H * W_)
    w = tl.load(W + c, mask=None).to(tl.float32)
    b = tl.load(B + c, mask=None).to(tl.float32)
    y = (x - mean) / tl.sqrt(var + EPS) * w + b
    tl.store(Y + c * N * H * W_ + offs, y, mask=mask)


class FusedInstanceNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        x = x.contiguous()
        y = torch.empty_like(x)
        N, C, H, W_ = x.shape
        BLOCK = N * H * W_
        _instance_norm_fwd[(C,)](x, y, weight, bias, N, C, H, W_, eps, BLOCK=BLOCK, num_warps=4)
        return y


def kernel_fn(x, weight, bias, eps=1e-5):
    return FusedInstanceNorm.apply(x, weight, bias, eps)


def can_use_kernel(x, weight, bias):
    return (x.is_cuda and
            x.is_contiguous() and
            x.dim() == 4 and
            x.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"x": (2, 256, 64, 64), "C": 256},
    "vit_h":  {"x": (2, 512, 32, 32), "C": 512},
    "small":  {"x": (8, 128, 32, 32), "C": 128},
}
