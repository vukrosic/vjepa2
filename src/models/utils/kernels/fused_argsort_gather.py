"""Fused argsort + gather kernel.

Source: src/models/predictor.py:239-241
Pattern:
    argsort = torch.argsort(masks, dim=1)  # [B, N]
    masks = torch.gather(masks, dim=1, index=argsort)
    x = torch.gather(x, dim=1, index=argsort.unsqueeze(-1).expand(-1, -1, x.size(-1)))
Fuses: the two gather operations into one kernel (the argsort stays in PyTorch as it needs sorting).
The kernel does: for each (b, row, d), gather x at sorted position, avoiding the expand().
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, argsort):
    """
    x: [B, N, D], argsort: [B, N] (sorted indices)
    Returns: x reordered by argsort along dim=1.
    """
    B, N, D = x.shape
    # PyTorch gather: need to expand argsort to [B, N, D]
    argsort_expanded = argsort.unsqueeze(-1).expand(B, N, D)
    return torch.gather(x, 1, argsort_expanded)


# --- KERNEL ---
@triton.jit
def _argsort_gather_kernel(
    X, IDX, OUT,
    B, N, D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (B * N,)
    Each program handles one row of D features.
    pid = b * N + n
    Loads sorted indices for the row, then for each d element, looks up the source index.
    """
    pid = tl.program_id(0)
    b = pid // N
    n = pid % N

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Load sorted indices for this row: IDX[b, :]
    # IDX layout: [B, N]
    # sorted_idx[b, n] = IDX[b*N + n]
    sorted_idx_n = tl.load(IDX + b * N + tl.arange(0, BLOCK_D), mask=mask_d, other=0).to(tl.int64)

    # For each d: x_out[b, n, d] = x_in[b, sorted_idx[d], d]
    # Wait, sorted_idx is per-row (N elements), not per-feature
    # All features use the same sorted index per row
    row_idx = tl.load(IDX + b * N + n).to(tl.int32)

    # Load the source row
    src_base = b * N * D + row_idx * D
    x_vals = tl.load(X + src_base + offs_d, mask=mask_d, other=0.0)
    tl.store(OUT + pid * D + offs_d, x_vals, mask=mask_d)


class FusedArgsortGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, argsort):
        B, N, D = x.shape
        BLOCK_D = triton.next_power_of_2(D)
        out = torch.empty_like(x)
        _argsort_gather_kernel[(B * N,)](
            x, argsort, out,
            B=B, N=N, D=D, BLOCK_D=BLOCK_D,
            num_warps=min(8, max(1, BLOCK_D // 32)),
        )
        ctx.save_for_backward(argsort)
        ctx.B = B; ctx.N = N; ctx.D = D
        return out

    @staticmethod
    def backward(ctx, grad_out):
        argsort, = ctx.saved_tensors
        B, N, D = ctx.B, ctx.N, ctx.D
        grad_x = torch.zeros_like(grad_out)

        @triton.jit
        def _scatter_kernel(GO, IDX, GX, B, N, D, BLOCK_D: tl.constexpr):
            pid = tl.program_id(0)
            b = pid // N
            n = pid % N
            row_idx = tl.load(IDX + b * N + n).to(tl.int32)
            offs_d = tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            go_vals = tl.load(GO + pid * D + offs_d, mask=mask_d, other=0.0)
            tl.store(GX + b * N * D + row_idx * D + offs_d, go_vals, mask=mask_d)

        BLOCK_D = triton.next_power_of_2(D)
        _scatter_kernel[(B * N,)](
            grad_out, argsort, grad_x,
            B, N, D, BLOCK_D=BLOCK_D,
            num_warps=min(8, max(1, BLOCK_D // 32)),
        )
        return grad_x, None


def kernel_fn(x, argsort):
    return FusedArgsortGather.apply(x, argsort)


def can_use_kernel(x, argsort):
    if not x.is_cuda:
        return False
    if x.ndim != 3 or argsort.ndim != 2:
        return False
    if x.shape[:2] != argsort.shape:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


SHAPES = {
    "small":  {"x": (2, 784, 384),  "D": 384},
    "medium": {"x": (2, 1568, 768), "D": 768},
    "large":  {"x": (2, 3136, 1024), "D": 1024},
}
