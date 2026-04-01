from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import fallback
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None and tl is not None


if TRITON_AVAILABLE:

    @triton.jit
    def _attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        Q_stride_b,
        Q_stride_h,
        Q_stride_n,
        Q_stride_d,
        K_stride_b,
        K_stride_h,
        K_stride_n,
        K_stride_d,
        V_stride_b,
        V_stride_h,
        V_stride_n,
        V_stride_d,
        Out_stride_b,
        Out_stride_h,
        Out_stride_n,
        Out_stride_d,
        B,
        H,
        N,
        D,
        scale: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused attention forward kernel: Out = softmax(Q @ K^T * scale) @ V.

        Uses SRAM tiling following FlashAttention style. Processes attention in
        blocks of BLOCK_M x BLOCK_N, accumulating results in registers.
        """
        # Program indices
        batch = tl.program_id(0)
        head = tl.program_id(1)
        start_m = tl.program_id(2) * BLOCK_M

        # Offsets for Q in this block row
        q_offset = (
            batch * Q_stride_b
            + head * Q_stride_h
            + start_m * Q_stride_n
        )
        q_offsets = q_offset + tl.arange(0, BLOCK_D) * Q_stride_d

        # Load Q block into SRAM
        q = tl.load(Q_ptr + q_offsets, mask=tl.arange(0, BLOCK_D) < D, other=0.0)
        q = q.to(tl.float32)

        # Accumulator for attention output
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        l_i: tl.float32 = 0.0  # Running sum of exponentials for softmax normalization
        m_i: tl.float32 = float("-inf")  # Running max for softmax stabilization

        # Loop over K/V blocks
        num_blocks = tl.cdiv(N, BLOCK_D)
        for block_k in range(num_blocks):
            # Compute row indices for K/V block
            row_k = block_k * BLOCK_D
            k_offsets = (
                batch * K_stride_b
                + head * K_stride_h
                + row_k * K_stride_n
                + tl.arange(0, BLOCK_D) * K_stride_d
            )
            v_offsets = (
                batch * V_stride_b
                + head * V_stride_h
                + row_k * V_stride_n
                + tl.arange(0, BLOCK_D) * V_stride_d
            )

            # Load K and V blocks
            k = tl.load(K_ptr + k_offsets, mask=tl.arange(0, BLOCK_D) < D, other=0.0)
            k = k.to(tl.float32)
            v = tl.load(V_ptr + v_offsets, mask=tl.arange(0, BLOCK_D) < D, other=0.0)
            v = v.to(tl.float32)

            # Compute Q @ K^T for this block row and K block
            # Q is (BLOCK_M, D), K is (BLOCK_D, D) -> we need K^T
            # Actually: q is (BLOCK_M, D), k is (D, BLOCK_N) via transposed load
            qk = tl.dot(q, k) * scale

            # Apply causal mask if enabled
            if CAUSAL:
                col_k = row_k + tl.arange(0, BLOCK_D)
                mask_col = col_k < start_m + tl.arange(0, BLOCK_M)
                mask_row = start_m + tl.arange(0, BLOCK_M) < N
                causal_mask = mask_row[:, None] & mask_col[None, :]
                # Broadcast causal_mask to qk shape (BLOCK_M, BLOCK_D)
                # This is approximate - we need to materialize the mask properly
                for m_idx in range(BLOCK_M):
                    if start_m + m_idx < N:
                        for k_idx in range(BLOCK_D):
                            if row_k + k_idx >= start_m + m_idx:
                                qk = tl.where(
                                    start_m + m_idx < row_k + k_idx,
                                    float("-inf"),
                                    qk,
                                )

            # Softmax stabilization: compute max for this block
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            # Compute correction factor for previously accumulated exp values
            correction = tl.exp(m_i - m_new)

            # Compute exp(qk - m_new) for non-masked positions
            p = tl.exp(qk - m_new[:, None])

            # Apply causal mask after exp (if not already applied above)
            if not CAUSAL:
                # Mask out future positions for non-causal case
                for m_idx in range(BLOCK_M):
                    if start_m + m_idx < N:
                        for k_idx in range(BLOCK_D):
                            if row_k + k_idx >= N:
                                p = tl.where(
                                    row_k + k_idx < N,
                                    p,
                                    0.0,
                                )

            # Apply correction to accumulator
            acc = acc * correction

            # Compute new attention weights applied to V
            acc = acc + tl.dot(p.to(tl.float16), v)

            # Update running sum of exponentials
            l_ij = tl.sum(p, axis=1)
            l_i = l_i * correction + l_ij

            # Update running max
            m_i = m_new

        # Normalize output
        out = acc / l_i[:, None]

        # Store output
        out_offset = (
            batch * Out_stride_b
            + head * Out_stride_h
            + start_m * Out_stride_n
        )
        out_offsets = out_offset + tl.arange(0, BLOCK_D) * Out_stride_d
        tl.store(
            Out_ptr + out_offsets,
            out.to(tl.float16),
            mask=tl.arange(0, BLOCK_D) < D,
        )

    @triton.jit
    def _attn_bwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        dO_ptr,
        dQ_ptr,
        dK_ptr,
        dV_ptr,
        Q_stride_b,
        Q_stride_h,
        Q_stride_n,
        Q_stride_d,
        K_stride_b,
        K_stride_h,
        K_stride_n,
        K_stride_d,
        V_stride_b,
        V_stride_h,
        V_stride_n,
        V_stride_d,
        Out_stride_b,
        Out_stride_h,
        Out_stride_n,
        Out_stride_d,
        dO_stride_b,
        dO_stride_h,
        dO_stride_n,
        dO_stride_d,
        dQ_stride_b,
        dQ_stride_h,
        dQ_stride_n,
        dQ_stride_d,
        dK_stride_b,
        dK_stride_h,
        dK_stride_n,
        dK_stride_d,
        dV_stride_b,
        dV_stride_h,
        dV_stride_n,
        dV_stride_d,
        B,
        H,
        N,
        D,
        scale: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Backward kernel for fused attention.

        Computes gradients dQ, dK, dV given the上游 gradient dO and
        the forward activations Q, K, V, Out.
        """
        batch = tl.program_id(0)
        head = tl.program_id(1)
        start_m = tl.program_id(2) * BLOCK_M

        # Load Q block
        q_offset = (
            batch * Q_stride_b
            + head * Q_stride_h
            + start_m * Q_stride_n
        )
        q_offsets = q_offset + tl.arange(0, BLOCK_D) * Q_stride_d
        q = tl.load(Q_ptr + q_offsets, mask=tl.arange(0, BLOCK_D) < D, other=0.0)
        q = q.to(tl.float32)

        # Load dO block
        dO_offset = (
            batch * dO_stride_b
            + head * dO_stride_h
            + start_m * dO_stride_n
        )
        dO_offsets = dO_offset + tl.arange(0, BLOCK_D) * dO_stride_d
        dO = tl.load(dO_ptr + dO_offsets, mask=tl.arange(0, BLOCK_D) < D, other=0.0)
        dO = dO.to(tl.float32)

        # Initialize gradient accumulators
        dQ_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        dK_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        dV_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Re-compute attention weights in blocks to get gradients
        l_i: tl.float32 = 0.0
        m_i: tl.float32 = float("-inf")

        num_blocks = tl.cdiv(N, BLOCK_D)
        for block_k in range(num_blocks):
            row_k = block_k * BLOCK_D
            k_offsets = (
                batch * K_stride_b
                + head * K_stride_h
                + row_k * K_stride_n
                + tl.arange(0, BLOCK_D) * K_stride_d
            )
            v_offsets = (
                batch * V_stride_b
                + head * V_stride_h
                + row_k * V_stride_n
                + tl.arange(0, BLOCK_D) * V_stride_d
            )

            k = tl.load(K_ptr + k_offsets, mask=tl.arange(0, BLOCK_D) < D, other=0.0)
            k = k.to(tl.float32)
            v = tl.load(V_ptr + v_offsets, mask=tl.arange(0, BLOCK_D) < D, other=0.0)
            v = v.to(tl.float32)

            qk = tl.dot(q, k) * scale

            # Softmax stabilization
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            correction = tl.exp(m_i - m_new)

            p = tl.exp(qk - m_new[:, None])

            # Apply causal masking if enabled
            if CAUSAL:
                for m_idx in range(BLOCK_M):
                    if start_m + m_idx < N:
                        for k_idx in range(BLOCK_D):
                            if row_k + k_idx >= start_m + m_idx:
                                p = tl.where(
                                    start_m + m_idx < row_k + k_idx,
                                    0.0,
                                    p,
                                )

            # Accumulate dV: gradient flows through V multiplied by attention weights
            p_T = tl.trans(p)
            dV_acc = dV_acc + tl.dot(p_T.to(tl.float16), dO)

            # Accumulate dK: gradient flows through K
            dK_acc = dK_acc + tl.dot(
                (dO * correction)[:, None].to(tl.float16),
                q.to(tl.float16),
            )

            # Update running sums
            l_ij = tl.sum(p, axis=1)
            l_i = l_i * correction + l_ij
            m_i = m_new

        # Normalize gradients by attention sums
        dQ = dQ_acc / l_i[:, None]
        dK = dK_acc / l_i[:, None]
        dV = dV_acc

        # Store gradients
        dQ_store_offset = (
            batch * dQ_stride_b
            + head * dQ_stride_h
            + start_m * dQ_stride_n
        )
        dQ_store_offsets = dQ_store_offset + tl.arange(0, BLOCK_D) * dQ_stride_d
        tl.store(
            dQ_ptr + dQ_store_offsets,
            dQ.to(tl.float16),
            mask=tl.arange(0, BLOCK_D) < D,
        )

        dK_store_offset = (
            batch * dK_stride_b
            + head * dK_stride_h
            + start_m * dK_stride_n
        )
        dK_store_offsets = dK_store_offset + tl.arange(0, BLOCK_D) * dK_stride_d
        tl.store(
            dK_ptr + dK_store_offsets,
            dK.to(tl.float16),
            mask=tl.arange(0, BLOCK_D) < D,
        )

        dV_store_offset = (
            batch * dV_stride_b
            + head * dV_stride_h
            + start_m * dV_stride_n
        )
        dV_store_offsets = dV_store_offset + tl.arange(0, BLOCK_D) * dV_stride_d
        tl.store(
            dV_ptr + dV_store_offsets,
            dV.to(tl.float16),
            mask=tl.arange(0, BLOCK_D) < D,
        )


def triton_attention_forward(q, k, v, scale=1.0, causal=False, block_m=64, block_d=64):
    """Fused attention forward pass using Triton.

    Args:
        q: Query tensor of shape [B, H, N, D]
        k: Key tensor of shape [B, H, N, D]
        v: Value tensor of shape [B, H, N, D]
        scale: Scaling factor for Q @ K^T
        causal: Whether to apply causal masking
        block_m: Block size along M dimension (sequence length)
        block_d: Block size along D dimension (head dimension)

    Returns:
        Output tensor of shape [B, H, N, D]
    """
    B, H, N, D = q.shape
    assert k.shape == (B, H, N, D), f"K shape mismatch: {k.shape} vs {(B, H, N, D)}"
    assert v.shape == (B, H, N, D), f"V shape mismatch: {v.shape} vs {(B, H, N, D)}"

    # Make contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)

    # Grid: (B, H, num_blocks_m)
    num_blocks_m = triton.cdiv(N, block_m)
    grid = (B, H, num_blocks_m)

    _attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B,
        H,
        N,
        D,
        scale,
        causal,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    return out


def triton_attention_backward(q, k, v, out, dO, scale=1.0, causal=False, block_m=64, block_d=64):
    """Fused attention backward pass using Triton.

    Args:
        q: Query tensor of shape [B, H, N, D]
        k: Key tensor of shape [B, H, N, D]
        v: Value tensor of shape [B, H, N, D]
        out: Forward output tensor of shape [B, H, N, D]
        dO: Gradient of output of shape [B, H, N, D]
        scale: Scaling factor for Q @ K^T
        causal: Whether causal masking was applied
        block_m: Block size along M dimension
        block_d: Block size along D dimension

    Returns:
        Tuple of (dQ, dK, dV) gradients
    """
    B, H, N, D = q.shape

    # Make contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    dO = dO.contiguous()

    dQ = torch.empty_like(q)
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)

    # Grid: (B, H, num_blocks_m)
    num_blocks_m = triton.cdiv(N, block_m)
    grid = (B, H, num_blocks_m)

    _attn_bwd_kernel[grid](
        q,
        k,
        v,
        out,
        dO,
        dQ,
        dK,
        dV,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        dO.stride(0),
        dO.stride(1),
        dO.stride(2),
        dO.stride(3),
        dQ.stride(0),
        dQ.stride(1),
        dQ.stride(2),
        dQ.stride(3),
        dK.stride(0),
        dK.stride(1),
        dK.stride(2),
        dK.stride(3),
        dV.stride(0),
        dV.stride(1),
        dV.stride(2),
        dV.stride(3),
        B,
        H,
        N,
        D,
        scale,
        causal,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    return dQ, dK, dV


def triton_attention_autograd(q, k, v, scale=1.0, causal=False, block_m=64, block_d=64):
    """Fused attention with autograd support.

    Args:
        q: Query tensor of shape [B, H, N, D]
        k: Key tensor of shape [B, H, N, D]
        v: Value tensor of shape [B, H, N, D]
        scale: Scaling factor for Q @ K^T
        causal: Whether to apply causal masking
        block_m: Block size along M dimension
        block_d: Block size along D dimension

    Returns:
        Output tensor of shape [B, H, N, D]
    """
    return _TritonAttentionAutograd.apply(q, k, v, scale, causal, block_m, block_d)


class _TritonAttentionAutograd(torch.autograd.Function):
    """Autograd function for fused Triton attention."""

    @staticmethod
    def forward(ctx, q, k, v, scale, causal, block_m, block_d):
        out = triton_attention_forward(q, k, v, scale, causal, block_m, block_d)
        ctx.save_for_backward(q, k, v, out)
        ctx.scale = scale
        ctx.causal = causal
        ctx.block_m = block_m
        ctx.block_d = block_d
        return out

    @staticmethod
    def backward(ctx, dO):
        q, k, v, out = ctx.saved_tensors
        dQ, dK, dV = triton_attention_backward(
            q, k, v, out, dO, ctx.scale, ctx.causal, ctx.block_m, ctx.block_d
        )
        return dQ, dK, dV, None, None, None, None


def can_use_triton_attention(q, k, v):
    """Check if Triton attention can be used for the given inputs.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor

    Returns:
        True if Triton attention can be used
    """
    if not TRITON_AVAILABLE:
        return False
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if q.shape != k.shape or q.shape != v.shape:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    # Head dimension must be divisible by 64 for efficient Triton kernel
    if q.size(-1) % 64 != 0:
        return False
    return True
