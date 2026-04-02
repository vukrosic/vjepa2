"""Fused Huber-Loss kernel.

Pattern: loss = mean(smooth_l1_loss(pred, target, beta=1.0))
  if |d| <= beta: 0.5 * d^2 / beta
  else: |d| - 0.5 * beta
Pure scalar loads.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(pred, target, beta=1.0):
    return torch.nn.functional.smooth_l1_loss(pred, target, beta=beta)


# --- KERNEL ---
@triton.jit
def _huber_loss_kernel(PRED, TARGET, OUT, N: tl.constexpr, BETA: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    local_sum = 0.0
    count = 0
    for i in range(BLOCK):
        offs = row_base + i
        if offs >= N:
            break
        p = tl.load(PRED + offs).to(tl.float32)
        t = tl.load(TARGET + offs).to(tl.float32)
        d = p - t
        abs_d = d if d >= 0 else -d
        if abs_d <= BETA:
            val = 0.5 * d * d / BETA
        else:
            val = abs_d - 0.5 * BETA
        local_sum = local_sum + val
        count = count + 1
    norm = local_sum / tl.cast(tl.maximum(count, 1), tl.float32)
    tl.store(OUT + pid, norm)


def kernel_fn(pred, target, beta=1.0):
    assert pred.is_contiguous() and target.is_contiguous()
    assert pred.shape == target.shape
    N = pred.numel()
    BLOCK = 1024
    n_blocks = (N + BLOCK - 1) // BLOCK
    partial = torch.empty(n_blocks, dtype=torch.float32, device=pred.device)
    _huber_loss_kernel[(n_blocks,)](pred, target, partial, N, beta, BLOCK=BLOCK, num_warps=4)
    return partial.sum() / N


def can_use_kernel(pred, target, beta=1.0):
    return (pred.is_cuda and target.is_cuda and
            pred.is_contiguous() and target.is_contiguous() and
            pred.shape == target.shape and
            pred.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l":  {"pred": (2, 1024, 4096), "target": (2, 1024, 4096)},
    "vit_h":  {"pred": (2, 2048, 5120), "target": (2, 2048, 5120)},
    "small":  {"pred": (8, 256, 1536), "target": (8, 256, 1536)},
}
