# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton kernels for optimized mask-related operations in V-JEPA 2."""

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
    def triton_apply_masks_2d_kernel(
        x_ptr,
        mask_ptr,
        out_ptr,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_mask_b,
        stride_mask_k,
        stride_out_b,
        stride_out_k,
        stride_out_d,
        B,
        N,
        D,
        K,
        BLOCK_D: tl.constexpr,
    ):
        """Kernel for applying 2D masks to tensor x.

        Args:
            x: Input tensor of shape [B, N, D]
            mask: Integer mask of shape [B, K] with indices in [0, N)
            Output: [B, K, D] tensor with gathered values
        """
        b = tl.program_id(0)
        k = tl.program_id(1)
        d = tl.program_id(2)

        cols = d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_cols = cols < D

        # Load mask indices for this batch and k position
        mask_idx = tl.load(mask_ptr + b * stride_mask_b + k * stride_mask_k)

        # Compute input offset using the mask index for N dimension
        x_offset = b * stride_x_b + mask_idx * stride_x_n + cols * stride_x_d

        # Load from input
        x_vals = tl.load(x_ptr + x_offset, mask=mask_cols, other=0.0)

        # Compute output offset
        out_offset = b * stride_out_b + k * stride_out_k + cols * stride_out_d

        # Store result
        tl.store(out_ptr + out_offset, x_vals, mask=mask_cols)

    @triton.jit
    def triton_apply_masks_1d_kernel(
        x_ptr,
        mask_ptr,
        out_ptr,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_mask_k,
        stride_out_b,
        stride_out_k,
        stride_out_d,
        B,
        N,
        D,
        K,
        BLOCK_D: tl.constexpr,
    ):
        """Kernel for applying 1D masks to tensor x.

        Args:
            x: Input tensor of shape [B, N, D]
            mask: Integer mask of shape [K] with indices in [0, N) - broadcast across batch
            Output: [B, K, D] tensor with gathered values
        """
        b = tl.program_id(0)
        k = tl.program_id(1)
        d = tl.program_id(2)

        cols = d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_cols = cols < D

        # Load mask index for this k position (same for all batch elements)
        mask_idx = tl.load(mask_ptr + k * stride_mask_k)

        # Compute input offset using the mask index for N dimension
        x_offset = b * stride_x_b + mask_idx * stride_x_n + cols * stride_x_d

        # Load from input
        x_vals = tl.load(x_ptr + x_offset, mask=mask_cols, other=0.0)

        # Compute output offset
        out_offset = b * stride_out_b + k * stride_out_k + cols * stride_out_d

        # Store result
        tl.store(out_ptr + out_offset, x_vals, mask=mask_cols)

    @triton.jit
    def triton_apply_masks_2d_batched_kernel(
        x_ptr,
        masks_ptr,
        out_ptr,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_masks_m,
        stride_masks_b,
        stride_masks_k,
        stride_out_m,
        stride_out_b,
        stride_out_k,
        stride_out_d,
        B,
        N,
        D,
        K,
        M,
        BLOCK_D: tl.constexpr,
    ):
        """Batched kernel for applying multiple 2D masks to tensor x.

        Args:
            x: Input tensor of shape [B, N, D]
            masks: Integer masks of shape [M, B, K] with M masks, indices in [0, N)
            Output: [M, B, K, D] tensor with gathered values
        """
        pid = tl.program_id(0)
        d = tl.program_id(1)

        k = pid % K
        tmp = pid // K
        b = tmp % B
        m = tmp // B

        cols = d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_cols = cols < D

        # Load mask indices for this mask, batch, and k position
        mask_idx = tl.load(
            masks_ptr + m * stride_masks_m + b * stride_masks_b + k * stride_masks_k
        )

        # Compute input offset using the mask index for N dimension
        x_offset = b * stride_x_b + mask_idx * stride_x_n + cols * stride_x_d

        # Load from input
        x_vals = tl.load(x_ptr + x_offset, mask=mask_cols, other=0.0)

        # Compute output offset
        out_offset = (
            m * stride_out_m + b * stride_out_b + k * stride_out_k + cols * stride_out_d
        )

        # Store result
        tl.store(out_ptr + out_offset, x_vals, mask=mask_cols)

    @triton.jit
    def triton_apply_masks_1d_batched_kernel(
        x_ptr,
        masks_ptr,
        out_ptr,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_masks_m,
        stride_masks_k,
        stride_out_m,
        stride_out_b,
        stride_out_k,
        stride_out_d,
        B,
        N,
        D,
        K,
        M,
        BLOCK_D: tl.constexpr,
    ):
        """Batched kernel for applying multiple 1D masks to tensor x.

        Args:
            x: Input tensor of shape [B, N, D]
            masks: Integer masks of shape [M, K] with M masks, indices in [0, N) - broadcast across batch
            Output: [M, B, K, D] tensor with gathered values
        """
        pid = tl.program_id(0)
        d = tl.program_id(1)

        k = pid % K
        tmp = pid // K
        b = tmp % B
        m = tmp // B

        cols = d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_cols = cols < D

        # Load mask index for this mask and k position
        mask_idx = tl.load(masks_ptr + m * stride_masks_m + k * stride_masks_k)

        # Compute input offset using the mask index for N dimension
        x_offset = b * stride_x_b + mask_idx * stride_x_n + cols * stride_x_d

        # Load from input
        x_vals = tl.load(x_ptr + x_offset, mask=mask_cols, other=0.0)

        # Compute output offset
        out_offset = (
            m * stride_out_m + b * stride_out_b + k * stride_out_k + cols * stride_out_d
        )

        # Store result
        tl.store(out_ptr + out_offset, x_vals, mask=mask_cols)


def can_use_triton_apply_masks(x, masks):
    """Check if triton kernels can be used for apply_masks operation."""
    if not TRITON_AVAILABLE or not x.is_cuda:
        return False
    if x.ndim != 3:
        return False
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return False
    if not isinstance(masks, (list, tuple)) or len(masks) == 0:
        return False
    # Check all masks have consistent shape
    first_mask = masks[0]
    if first_mask.ndim == 1:
        # 1D masks: all should be same shape [K]
        K = first_mask.numel()
        if not all(m.ndim == 1 and m.shape == first_mask.shape for m in masks):
            return False
        return True
    elif first_mask.ndim == 2:
        # 2D masks: all should be same shape [B, K]
        if not all(m.ndim == 2 and m.shape == first_mask.shape for m in masks):
            return False
        return True
    return False


def triton_apply_masks(x, masks, concat=True):
    """Apply masks to tensor x using optimized Triton kernel.

    Args:
        x: Input tensor of shape [B, N, D]
        masks: List of integer masks. Each mask is either [K] (1D) or [B, K] (2D)
               containing indices into dimension N.
        concat: If True, concatenate results along batch dimension to [M*K, K, D] or [M*B, K, D].
                If False, return list of tensors.

    Returns:
        Gathered tensor or list of gathered tensors.
    """
    if not can_use_triton_apply_masks(x, masks):
        # Fall back to torch implementation
        from src.masks.utils import apply_masks as torch_apply_masks

        return torch_apply_masks(x, masks, concat=concat)

    B, N, D = x.shape
    first_mask = masks[0]

    if first_mask.ndim == 1:
        # 1D case: masks are [K], broadcast across batch
        K = first_mask.numel()
        M = len(masks)
        masks_tensor = torch.stack(masks, dim=0)  # [M, K]

        out = torch.empty((M, B, K, D), dtype=x.dtype, device=x.device)
        BLOCK_D = min(128, triton.next_power_of_2(D))

        grid = (M * B * K, triton.cdiv(D, BLOCK_D))
        triton_apply_masks_1d_batched_kernel[grid](
            x,
            masks_tensor,
            out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            masks_tensor.stride(0),
            masks_tensor.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            N,
            D,
            K,
            M,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )

        if not concat:
            return list(out.unbind(0))
        return out.reshape(-1, K, D)

    else:
        # 2D case: masks are [B, K]
        K = first_mask.shape[1]
        M = len(masks)
        masks_tensor = torch.stack(masks, dim=0)  # [M, B, K]

        out = torch.empty((M, B, K, D), dtype=x.dtype, device=x.device)
        BLOCK_D = min(128, triton.next_power_of_2(D))

        grid = (M * B * K, triton.cdiv(D, BLOCK_D))
        triton_apply_masks_2d_batched_kernel[grid](
            x,
            masks_tensor,
            out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            masks_tensor.stride(0),
            masks_tensor.stride(1),
            masks_tensor.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            N,
            D,
            K,
            M,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )

        if not concat:
            return list(out.unbind(0))
        return out.reshape(-1, K, D)


def triton_apply_masks_single(x, mask):
    """Apply a single mask to tensor x using optimized Triton kernel.

    Args:
        x: Input tensor of shape [B, N, D]
        mask: Integer mask of shape [K] (1D) or [B, K] (2D) containing indices into dimension N.

    Returns:
        Gathered tensor of shape [B, K, D].
    """
    if not TRITON_AVAILABLE or not x.is_cuda or x.ndim != 3:
        from src.masks.utils import apply_masks as torch_apply_masks

        return torch_apply_masks(x, [mask], concat=True)

    B, N, D = x.shape

    if mask.ndim == 1:
        K = mask.numel()
        out = torch.empty((B, K, D), dtype=x.dtype, device=x.device)
        BLOCK_D = min(128, triton.next_power_of_2(D))

        grid = (B, K, triton.cdiv(D, BLOCK_D))
        triton_apply_masks_1d_kernel[grid](
            x,
            mask,
            out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            mask.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            B,
            N,
            D,
            K,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )
    else:
        K = mask.shape[1]
        out = torch.empty((B, K, D), dtype=x.dtype, device=x.device)
        BLOCK_D = min(128, triton.next_power_of_2(D))

        grid = (B, K, triton.cdiv(D, BLOCK_D))
        triton_apply_masks_2d_kernel[grid](
            x,
            mask,
            out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            mask.stride(0),
            mask.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            B,
            N,
            D,
            K,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )

    return out
