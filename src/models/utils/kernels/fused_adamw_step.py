"""Fused AdamW optimizer step kernel.

Source: Training loop optimizer step
Pattern: moment1 = beta1 * m + (1-beta1) * g; moment2 = beta2 * v + (1-beta2) * g^2;
         m_hat = m / (1 - beta1^t); v_hat = v / (1 - beta2^t)
         param = param * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)
Fuses: 3 reads (param, m, v) + gradient -> 3 writes (param, m, v) in one kernel.
Without fusion: 6 separate passes over parameter memory.
Frequency: Every training iteration, for all parameters.
"""
import math
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy from PyTorch AdamW logic) ---
def baseline_fn(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step):
    """Single AdamW step (in-place)."""
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    # Weight decay (decoupled)
    param.mul_(1.0 - lr * weight_decay)

    # Update moments
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    # Bias-corrected estimates
    step_size = lr / bias_correction1
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    param.addcdiv_(exp_avg, denom, value=-step_size)
    return param, exp_avg, exp_avg_sq


# --- KERNEL ---
@triton.jit
def _adamw_step_kernel(
    PARAM_ptr, GRAD_ptr, EXP_AVG_ptr, EXP_AVG_SQ_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    lr: tl.constexpr,
    beta1: tl.constexpr,
    beta2: tl.constexpr,
    eps: tl.constexpr,
    weight_decay: tl.constexpr,
    bias_correction1: tl.constexpr,
    bias_correction2: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load all in one pass (3 reads)
    p = tl.load(PARAM_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(GRAD_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(EXP_AVG_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(EXP_AVG_SQ_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Weight decay
    p = p * (1.0 - lr * weight_decay)

    # Update moments
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g

    # Bias-corrected update
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2
    p = p - lr * m_hat / (tl.sqrt(v_hat) + eps)

    # Store all in one pass (3 writes)
    tl.store(PARAM_ptr + offs, p.to(p.dtype), mask=mask)
    tl.store(EXP_AVG_ptr + offs, m.to(m.dtype), mask=mask)
    tl.store(EXP_AVG_SQ_ptr + offs, v.to(v.dtype), mask=mask)


def kernel_fn(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step):
    """Fused in-place AdamW step."""
    assert param.is_contiguous()
    assert grad.is_contiguous()
    assert exp_avg.is_contiguous()
    assert exp_avg_sq.is_contiguous()

    N = param.numel()
    BLOCK = 1024
    n_blocks = (N + BLOCK - 1) // BLOCK

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    _adamw_step_kernel[(n_blocks,)](
        param, grad, exp_avg, exp_avg_sq,
        N=N, BLOCK=BLOCK,
        lr=lr, beta1=beta1, beta2=beta2, eps=eps,
        weight_decay=weight_decay,
        bias_correction1=bc1,
        bias_correction2=bc2,
        num_warps=4,
    )
    return param, exp_avg, exp_avg_sq


def can_use_kernel(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step):
    return (param.is_cuda and
            all(t.is_contiguous() for t in [param, grad, exp_avg, exp_avg_sq]) and
            param.shape == grad.shape == exp_avg.shape == exp_avg_sq.shape and
            param.dtype in (torch.float16, torch.bfloat16, torch.float32))


# Realistic: typical weight matrices in ViT-L/H
SHAPES = {
    "attn_proj":   {"param": (1024, 1024), "N": 1024 * 1024},
    "mlp_fc":      {"param": (4096, 1024), "N": 4096 * 1024},
    "vit_h_attn":  {"param": (1280, 1280), "N": 1280 * 1280},
}
