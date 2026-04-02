"""Fused Binary Cross Entropy kernel.

Pattern: loss = -sum(target * log(pred + eps) + (1 - target) * log(1 - pred + eps)) / N
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(pred, target):
    return torch.nn.functional.binary_cross_entropy(pred, target)


@triton.jit
def _bce_fwd(Pred, Target, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    total = 0.0
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        p = tl.load(Pred + offs).to(tl.float32)
        t = tl.load(Target + offs).to(tl.float32)
        p_clamp = tl.minimum(tl.maximum(p, EPS), 1.0 - EPS)
        total += -t * tl.log(p_clamp) - (1.0 - t) * tl.log(1.0 - p_clamp)
    result = total / N
    tl.store(Y, result)


class FusedBinaryCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target):
        assert pred.is_contiguous() and target.is_contiguous()
        ctx.save_for_backward(pred, target)
        y = torch.empty(1, dtype=torch.float32, device=pred.device)
        N = pred.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        _bce_fwd[grid](pred, target, y, N, 1e-7, BLOCK=BLOCK, num_warps=4)
        return y


def kernel_fn(pred, target):
    return FusedBinaryCrossEntropy.apply(pred, target)


def can_use_kernel(pred, target):
    return (pred.is_cuda and target.is_cuda and
            pred.is_contiguous() and target.is_contiguous() and
            pred.shape == target.shape and
            pred.dtype in (torch.float16, torch.bfloat16, torch.float32))


SHAPES = {
    "vit_l":  {"pred": (2, 1024, 4096), "target": (2, 1024, 4096)},
    "vit_h":  {"pred": (2, 2048, 5120), "target": (2, 2048, 5120)},
    "small":  {"pred": (8, 256, 1536), "target": (8, 256, 1536)},
}
