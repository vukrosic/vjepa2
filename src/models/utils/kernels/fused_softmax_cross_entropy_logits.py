"""Fused softmax + cross entropy kernel for distillation loss.

Pattern: same as F.cross_entropy(logits / T, target, reduction='mean')
Fuses: subtract max + exp + sum + log-softmax in one kernel pass.
Numerically stable: avoids intermediate softmax probabilities tensor.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(logits, target_indices, temperature=0.07):
    """Same as: F.cross_entropy(logits / T, target, reduction='mean')"""
    return torch.nn.functional.cross_entropy(logits / temperature, target_indices)


# --- KERNEL ---
@triton.jit
def _fused_softmax_ce_fwd(LOGITS, TARGETS, Y, B: tl.constexpr, C: tl.constexpr, TEMP: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_b = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    row_base = pid_b * C

    # Phase 1: subtract max per row (numerical stability)
    max_val = -1e9
    for c in range(C):
        x = tl.load(LOGITS + row_base + c, mask=mask_c, other=0.0).to(tl.float32)
        max_val = tl.max(max_val, x / TEMP)
        mask_c = (offs_c + c + 1) < C
    mask_c = offs_c < C

    # Phase 2: compute exp sum
    exp_sum = 0.0
    for c in range(C):
        x = tl.load(LOGITS + row_base + c, mask=mask_c, other=0.0).to(tl.float32)
        exp_sum += tl.exp(x / TEMP - max_val)
        mask_c = (offs_c + c + 1) < C
    mask_c = offs_c < C

    # Phase 3: compute cross-entropy loss: -log(exp(x_t)/sum) = log(sum) - x_t
    target = tl.load(TARGETS + pid_b).to(tl.int32)
    x_target = tl.load(LOGITS + row_base + target, mask=mask_c, other=0.0).to(tl.float32)
    loss = max_val + tl.log(exp_sum) - x_target / TEMP
    tl.store(Y + pid_b, loss)


def kernel_fn(logits, target_indices, temperature=0.07):
    assert logits.is_contiguous() and target_indices.is_contiguous()
    B, C = logits.shape
    y = torch.empty(B, dtype=logits.dtype, device=logits.device)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (B,)
    _fused_softmax_ce_fwd[grid](logits, target_indices, y, B, C, temperature, BLOCK_C=BLOCK_C, num_warps=4)
    return y.mean()


def can_use_kernel(logits, target_indices, temperature):
    return (logits.is_cuda and target_indices.is_cuda and
            logits.is_contiguous() and target_indices.is_contiguous() and
            logits.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            target_indices.dtype == torch.long)


SHAPES = {
    "small":   {"logits": (256, 512),   "target_shape": (256,)},
    "medium":  {"logits": (512, 1024),  "target_shape": (512,)},
    "large":   {"logits": (1024, 2048), "target_shape": (1024,)},
}
