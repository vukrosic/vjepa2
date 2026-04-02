"""Fused Causal Mask + RoPE + Softmax kernel.

Pattern: attention_scores = softmax_masked(Q_rotated @ K_rotated^T / sqrt(D), causal_mask)
Fuses: RoPE rotation + QK computation + causal masking + softmax into one pass.
Causal mask ensures position j only attends to positions i <= j.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(q, k, cos_table, sin_table):
    """Apply RoPE to Q and K, then compute masked softmax attention scores."""
    import math
    B, H, T, D = q.shape
    D_half = D // 2
    # Apply RoPE
    q_rot = q.clone()
    k_rot = k.clone()
    q_rot[..., :D_half] = q[..., :D_half] * cos_table - q[..., D_half:] * sin_table
    q_rot[..., D_half:] = q[..., :D_half] * sin_table + q[..., D_half:] * cos_table
    k_rot[..., :D_half] = k[..., :D_half] * cos_table - k[..., D_half:] * sin_table
    k_rot[..., D_half:] = k[..., :D_half] * sin_table + k[..., D_half:] * cos_table
    # QK^T / sqrt(D)
    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum('bhmd,bhnd->bhmn', q_rot, k_rot) * scale
    # Apply causal mask (lower triangular)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~mask, float('-inf'))
    # Softmax
    scores_softmax = torch.nn.functional.softmax(scores, dim=-1)
    return scores_softmax


# --- KERNEL ---
@triton.jit
def _fused_masked_softmax_rope_fwd(
    Q, K, COS, SIN, Y,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
):
    """One program per (b, h, t). Compute RoPE + QK + causal softmax."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    D_half = D // 2

    # Q row base
    q_base = ((b * H + h) * T + t) * D

    # Online max over k positions j where j <= t (causal)
    m_val = -1e9
    for j in range(t + 1):
        k_base = ((b * H + h) * T + j) * D
        qk = 0.0
        for d in range(D):
            q_val = tl.load(Q + q_base + d).to(tl.float32)
            k_val = tl.load(K + k_base + d).to(tl.float32)
            if d < D_half:
                cos_d = tl.load(COS + d)
                sin_d = tl.load(SIN + d)
                # RoPE for Q: q_rot[d] = q[d] * cos - q[d+D_half] * sin
                q_rot_d = q_val * cos_d - tl.load(Q + q_base + D_half + d) * sin_d
                # RoPE for K: k_rot[d] = k[d] * cos - k[d+D_half] * sin
                k_rot_d = k_val * cos_d - tl.load(K + k_base + D_half + d) * sin_d
                qk += q_rot_d * k_rot_d
            else:
                qk += q_val * k_val
        m_val = tl.where(m_val > qk, m_val, qk)

    # Online exp-sum over causal positions
    e_sum = 0.0
    for j in range(t + 1):
        k_base = ((b * H + h) * T + j) * D
        qk = 0.0
        for d in range(D):
            q_val = tl.load(Q + q_base + d).to(tl.float32)
            k_val = tl.load(K + k_base + d).to(tl.float32)
            if d < D_half:
                cos_d = tl.load(COS + d)
                sin_d = tl.load(SIN + d)
                q_rot_d = q_val * cos_d - tl.load(Q + q_base + D_half + d) * sin_d
                k_rot_d = k_val * cos_d - tl.load(K + k_base + D_half + d) * sin_d
                qk += q_rot_d * k_rot_d
            else:
                qk += q_val * k_val
        e_sum += tl.exp(qk - m_val)

    # Store softmax output for all causal k positions
    for j in range(t + 1):
        k_base = ((b * H + h) * T + j) * D
        qk = 0.0
        for d in range(D):
            q_val = tl.load(Q + q_base + d).to(tl.float32)
            k_val = tl.load(K + k_base + d).to(tl.float32)
            if d < D_half:
                cos_d = tl.load(COS + d)
                sin_d = tl.load(SIN + d)
                q_rot_d = q_val * cos_d - tl.load(Q + q_base + D_half + d) * sin_d
                k_rot_d = k_val * cos_d - tl.load(K + k_base + D_half + d) * sin_d
                qk += q_rot_d * k_rot_d
            else:
                qk += q_val * k_val
        softmax_val = tl.exp(qk - m_val) / e_sum
        y_base = ((b * H + h) * T + t) * T + j
        tl.store(Y + y_base, softmax_val)
        # Store zeros for masked positions (j > t)
        for j_fill in range(t + 1, T):
            y_fill_base = ((b * H + h) * T + t) * T + j_fill
            tl.store(Y + y_fill_base, 0.0)


class FusedMaskedSoftmaxRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos_table, sin_table):
        B, H, T, D = q.shape
        y = torch.zeros(B, H, T, T, dtype=q.dtype, device=q.device)
        _fused_masked_softmax_rope_fwd[(B, H, T)](
            q.contiguous(), k.contiguous(), cos_table.contiguous(), sin_table.contiguous(), y,
            B, H, T, D,
            num_warps=4,
        )
        ctx.save_for_backward(q, k, cos_table, sin_table)
        ctx.T = T
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, cos_table, sin_table = ctx.saved_tensors
        import math
        B, H, T, D = q.shape
        D_half = D // 2
        q_rot = q.clone()
        k_rot = k.clone()
        q_rot[..., :D_half] = q[..., :D_half] * cos_table - q[..., D_half:] * sin_table
        q_rot[..., D_half:] = q[..., :D_half] * sin_table + q[..., D_half:] * cos_table
        k_rot[..., :D_half] = k[..., :D_half] * cos_table - k[..., D_half:] * sin_table
        k_rot[..., D_half:] = k[..., :D_half] * sin_table + k[..., D_half:] * cos_table
        scale = 1.0 / math.sqrt(D)
        scores = torch.einsum('bhmd,bhnd->bhmn', q_rot, k_rot) * scale
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, float('-inf'))
        scores_softmax = torch.nn.functional.softmax(scores, dim=-1)
        grad = torch.autograd.grad(
            scores_softmax, (q, k), grad_outputs=(dy.contiguous(),)
        )
        return grad[0], grad[1], None, None


def kernel_fn(q, k, cos_table, sin_table):
    return FusedMaskedSoftmaxRoPE.apply(q, k, cos_table, sin_table)


def can_use_kernel(q, k, cos_table, sin_table):
    if not (q.is_cuda and k.is_cuda and cos_table.is_cuda and sin_table.is_cuda):
        return False
    if not q.is_contiguous():
        return False
    if q.shape != k.shape:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


SHAPES = {
    "vit_small": {"T": 256, "D": 64, "H": 6},
    "vit_l":     {"T": 1024, "D": 64, "H": 12},
    "vit_h":     {"T": 2048, "D": 80, "H": 16},
}
