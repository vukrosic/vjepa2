"""Fused gather + add kernel.

Source: V-JEPA 2 predictor — gathers selected context tokens and adds to predictions.
Pattern: gathered = x.gather(1, indices.unsqueeze(-1).expand(...)); accum + gathered
Fuses: gather + add into one kernel pass — avoids materializing the gathered tensor.
Frequency: Every predictor forward pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(x, indices, accum):
    """
    x: [B, N, D], indices: [B, M], accum: [B, M, D]
    Returns: accum + gather(x, 1, indices_expanded)
    """
    B, M = indices.shape
    gathered = torch.gather(x, 1, indices.unsqueeze(-1).expand(B, M, x.shape[-1]))
    return accum + gathered


# --- KERNEL ---
class FusedGatherAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, accum):
        B, M, D = x.shape[0], indices.shape[1], x.shape[-1]
        indices_flat = indices.flatten()
        gathered = torch.empty(B * M, D, dtype=x.dtype, device=x.device)

        @triton.jit
        def _gather_kernel(X, IDX, OUT, B, M, D, BLOCK_D):
            pid = tl.program_id(0)
            b = pid // M
            m = pid % M
            idx = tl.load(IDX + pid).to(tl.int32)
            x_base = b * M * D + idx * D
            out_base = pid * D
            for off in range(0, D, BLOCK_D):
                cols = off + tl.arange(0, BLOCK_D)
                mask = cols < D
                val = tl.load(X + x_base + cols, mask=mask, other=0.0)
                tl.store(OUT + out_base + cols, val, mask=mask)

        BLOCK_D = min(triton.next_power_of_2(D), 2048)
        _gather_kernel[(B * M,)](
            x, indices_flat, gathered,
            B, M, D, BLOCK_D=BLOCK_D,
            num_warps=min(8, max(1, BLOCK_D // 64)),
        )
        gathered = gathered.view(B, M, D)
        ctx.save_for_backward(x, indices_flat, accum)
        ctx.B = B; ctx.M = M; ctx.D = D
        return accum + gathered

    @staticmethod
    def backward(ctx, grad_out):
        x, indices_flat, accum = ctx.saved_tensors
        B, M, D = ctx.B, ctx.M, ctx.D
        grad_accum = grad_out.clone()
        grad_x = torch.zeros_like(x)

        @triton.jit
        def _gather_add_bwd(G, IDX, GX, B, M, D, BLOCK_D):
            pid = tl.program_id(0)
            b = pid // M
            m = pid % M
            idx = tl.load(IDX + pid).to(tl.int32)
            g_base = pid * D
            x_base = b * M * D + idx * D
            for off in range(0, D, BLOCK_D):
                cols = off + tl.arange(0, BLOCK_D)
                mask = cols < D
                g = tl.load(G + g_base + cols, mask=mask, other=0.0)
                tl.atomic_add(GX + x_base + cols, g, mask=mask)

        BLOCK_D = min(triton.next_power_of_2(D), 2048)
        grad_out_flat = grad_out.view(B * M, D)
        _gather_add_bwd[(B * M,)](
            grad_out_flat, indices_flat, grad_x,
            B, M, D, BLOCK_D=BLOCK_D,
            num_warps=min(8, max(1, BLOCK_D // 64)),
        )
        return grad_x, None, grad_accum


def kernel_fn(x, indices, accum):
    return FusedGatherAdd.apply(x, indices, accum)


def can_use_kernel(x, indices, accum):
    if not x.is_cuda:
        return False
    if x.ndim != 3 or indices.ndim != 2 or accum.ndim != 3:
        return False
    if x.shape[0] != indices.shape[0] or x.shape[0] != accum.shape[0]:
        return False
    if x.shape[1] < indices.shape[1] or x.shape[2] != accum.shape[2]:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


SHAPES = {
    "small":  {"x": (2, 784, 384),  "indices_shape": (2, 196), "D": 384},
    "medium": {"x": (2, 1568, 1024), "indices_shape": (2, 512), "D": 1024},
    "large":  {"x": (2, 3136, 1280), "indices_shape": (2, 1024), "D": 1280},
}
