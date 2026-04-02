"""Fused token scatter kernel.

Scatters M unmasked tokens back into a full token grid of size total_tokens.
This is the reverse of gather — used in the V-JEPA 2 predictor after
masked token prediction to reconstruct the full feature map.

Baseline uses torch.scatter_ with zero-fill. The Triton kernel does one
program per (B, M) token and copies D features in a single vectorized pass.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE ---
def baseline_fn(src, indices, total_tokens, embed_dim):
    """src: [B, M, D], indices: [B, M] — scatter M tokens into [B, total_tokens, D] (zeros elsewhere)."""
    B, M, D = src.shape
    out = torch.zeros(B, total_tokens, D, dtype=src.dtype, device=src.device)
    out.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, D), src)
    return out


# --- KERNEL ---
@triton.jit
def _token_scatter_fwd_kernel(
    SRC, IDX, OUT,
    B, M, D: tl.constexpr,
    total_tokens,
    BLOCK_D: tl.constexpr,
):
    """Grid: (B * M,)
    pid = b * M + m
    Flattens indices to 1D [B*M] so IDX[pid] correctly loads the dest index.
    """
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M

    idx = tl.load(IDX + pid).to(tl.int32)

    src_base = pid * D
    out_base = (b * total_tokens + idx) * D

    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        # Block-level pointer arithmetic
        src_ptrs = src_base + cols
        out_ptrs = out_base + cols
        val = tl.load(SRC + src_ptrs, mask=mask, other=0.0)
        tl.store(OUT + out_ptrs, val, mask=mask)


class FusedTokenScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, indices, total_tokens, embed_dim):
        B, M, D = src.shape
        indices_flat = indices.flatten()  # [B*M]
        ctx.save_for_backward(indices_flat)
        ctx.total_tokens = total_tokens
        ctx.B = B; ctx.M = M; ctx.D = D

        out = torch.zeros(B, total_tokens, D, dtype=src.dtype, device=src.device)
        BLOCK_D = min(triton.next_power_of_2(D), 2048)
        _token_scatter_fwd_kernel[(B * M,)](
            src, indices_flat, out,
            B=B, M=M, D=D,
            total_tokens=total_tokens,
            BLOCK_D=BLOCK_D,
            num_warps=min(8, max(1, BLOCK_D // 64)),
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        indices_flat, = ctx.saved_tensors
        B, M, D = ctx.B, ctx.M, ctx.D
        total_tokens = ctx.total_tokens

        grad_src = torch.zeros(B * M, D, dtype=grad_out.dtype, device=grad_out.device)

        @triton.jit
        def _scatter_bwd(G, IDX, GS, B, M, D: tl.constexpr, total_tokens, BLOCK_D: tl.constexpr):
            pid = tl.program_id(0)
            b = pid // M
            m = pid % M
            idx = tl.load(IDX + pid).to(tl.int32)
            src_base = pid * D
            out_base = (b * total_tokens + idx) * D
            for off in range(0, D, BLOCK_D):
                cols = off + tl.arange(0, BLOCK_D)
                mask = cols < D
                # Block-level pointer arithmetic
                out_ptrs = out_base + cols
                src_ptrs = src_base + cols
                g = tl.load(G + out_ptrs, mask=mask, other=0.0)
                tl.store(GS + src_ptrs, g, mask=mask)

        BLOCK_D = min(triton.next_power_of_2(D), 2048)
        _scatter_bwd[(B * M,)](
            grad_out, indices_flat, grad_src,
            B, M, D, total_tokens, BLOCK_D=BLOCK_D,
            num_warps=min(8, max(1, BLOCK_D // 64)),
        )
        return grad_src.view(B, M, D), None, None, None


def kernel_fn(src, indices, total_tokens, embed_dim):
    return FusedTokenScatter.apply(src, indices, total_tokens, embed_dim)


def can_use_kernel(src, indices, total_tokens, embed_dim):
    return (
        src.is_cuda and indices.is_cuda
        and src.is_contiguous() and indices.is_contiguous()
        and src.ndim == 3
        and indices.ndim == 2
        and src.shape[:2] == indices.shape
        and src.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


SHAPES = {
    "small_mask":  {"src": (2, 196, 384),  "total_tokens": 784},
    "medium_mask": {"src": (2, 512, 1024), "total_tokens": 1568},
    "large_mask":  {"src": (2, 1024, 1280), "total_tokens": 3136},
}
