"""Row-wise softmax kernel using a stable online-style reduction.

Computes softmax over the last dimension of a contiguous tensor by flattening
the leading dimensions into rows. One Triton program handles one row.
"""
import torch
import triton
import triton.language as tl


def baseline_fn(scores):
    """scores: [..., D] — softmax over the last dimension."""
    return torch.softmax(scores, dim=-1)


@triton.jit
def _online_softmax_fwd(X, Y, STRIDE_ROW, D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = X + pid * STRIDE_ROW
    out_ptr = Y + pid * STRIDE_ROW

    row_max = -float("inf")
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        vals = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))

    exp_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        vals = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        exp_acc += tl.where(mask, tl.exp(vals - row_max), 0.0)
    denom = tl.sum(exp_acc, axis=0)

    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        vals = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        probs = tl.exp(vals - row_max) / denom
        tl.store(out_ptr + cols, probs, mask=mask)


def kernel_fn(scores):
    if not can_use_kernel(scores):
        return baseline_fn(scores)

    scores_2d = scores.reshape(-1, scores.shape[-1]).contiguous()
    rows, d_model = scores_2d.shape
    out_2d = torch.empty_like(scores_2d)

    block_d = min(triton.next_power_of_2(d_model), 4096)
    num_warps = min(16, max(1, block_d // 32))
    _online_softmax_fwd[(rows,)](
        scores_2d,
        out_2d,
        scores_2d.stride(0),
        D=d_model,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out_2d.reshape_as(scores)


def can_use_kernel(scores):
    return (
        scores.is_cuda
        and scores.is_contiguous()
        and scores.ndim >= 1
        and scores.shape[-1] > 0
        and scores.shape[-1] <= 4096
        and scores.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


SHAPES = {
    "vit_l_short": {"scores": (2, 16, 256, 256)},
    "vit_l_medium": {"scores": (2, 16, 512, 512)},
    "vit_l_long": {"scores": (1, 16, 1024, 1024)},
}
