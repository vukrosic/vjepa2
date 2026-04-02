"""Fused cross-entropy loss with logit scaling.

Common pattern in contrastive learning (e.g., CLIP-style losses):
  loss = cross_entropy(logits / temperature, labels)

Fuses: divide-by-temperature + row-max + log-sum-exp + NLL into one kernel
launch per row, avoiding two full passes over the logits tensor.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact pattern from contrastive training) ---
def baseline_fn(logits, labels, temperature=0.07):
    """logits: [B, C], labels: [B] long ints."""
    return torch.nn.functional.cross_entropy(logits / temperature, labels)


# --- KERNEL ---
@triton.jit
def _fused_cross_entropy_fwd(
    LOGITS,
    LABELS,
    LOSSES,
    B: tl.constexpr,
    C: tl.constexpr,
    temperature: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One program per row (sample in batch)."""
    row = tl.program_id(0)
    if row >= B:
        return

    row_start = row * C
    label = tl.load(LABELS + row)

    # --- Pass 1: find row max (for numerical stability) ---
    row_max = -float("inf")
    for block_start in range(0, C, BLOCK_C):
        offs = block_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        logit = tl.load(LOGITS + row_start + offs, mask=mask, other=-float("inf")).to(tl.float32)
        logit = logit / temperature
        block_max = tl.max(logit, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # --- Pass 2: compute log-sum-exp and the label logit ---
    lse = 0.0
    label_logit = 0.0
    for block_start in range(0, C, BLOCK_C):
        offs = block_start + tl.arange(0, BLOCK_C)
        mask = offs < C
        logit = tl.load(LOGITS + row_start + offs, mask=mask, other=-float("inf")).to(tl.float32)
        logit = logit / temperature
        lse += tl.sum(tl.exp(logit - row_max) * mask.to(tl.float32), axis=0)
        # Accumulate label logit (only one entry is the true label)
        is_label = (offs == label) & mask
        label_logit += tl.sum(tl.where(is_label, logit, 0.0), axis=0)

    log_lse = tl.log(lse) + row_max
    loss = log_lse - label_logit
    tl.store(LOSSES + row, loss)


def kernel_fn(logits, labels, temperature=0.07):
    """Compute cross-entropy loss with fused temperature scaling."""
    assert logits.is_contiguous() and labels.is_contiguous()
    B, C = logits.shape
    losses = torch.empty(B, dtype=torch.float32, device=logits.device)
    BLOCK_C = triton.next_power_of_2(min(C, 1024))
    grid = (B,)
    _fused_cross_entropy_fwd[grid](
        logits.float(),
        labels,
        losses,
        B=B,
        C=C,
        temperature=temperature,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return losses.mean()


def can_use_kernel(logits, labels, temperature=0.07):
    return (
        logits.is_cuda
        and labels.is_cuda
        and logits.is_contiguous()
        and labels.is_contiguous()
        and logits.ndim == 2
        and labels.ndim == 1
        and labels.shape[0] == logits.shape[0]
        and logits.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and labels.dtype == torch.long
    )


SHAPES = {
    "small":  {"logits": (256, 256),   "labels_shape": (256,),   "temperature": 0.07},
    "medium": {"logits": (512, 512),   "labels_shape": (512,),   "temperature": 0.07},
    "large":  {"logits": (1024, 1024), "labels_shape": (1024,),  "temperature": 0.07},
}
