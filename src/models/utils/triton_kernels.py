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
    def rope_rotate_kernel(
        x_ptr,
        pos_ptr,
        omega_ptr,
        out_ptr,
        stride_x_b,
        stride_x_h,
        stride_x_n,
        stride_x_d,
        stride_out_b,
        stride_out_h,
        stride_out_n,
        stride_out_d,
        H,
        N,
        D,
        HALF_D,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        cols = block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = cols < D

        n = row % N
        tmp = row // N
        h = tmp % H
        b = tmp // H

        x_offsets = b * stride_x_b + h * stride_x_h + n * stride_x_n + cols * stride_x_d
        pair_cols = cols ^ 1
        pair_offsets = b * stride_x_b + h * stride_x_h + n * stride_x_n + pair_cols * stride_x_d

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        pair = tl.load(x_ptr + pair_offsets, mask=mask, other=0.0)
        y = tl.where((cols & 1) == 0, -pair, pair)

        omega_idx = cols % HALF_D
        omega = tl.load(omega_ptr + omega_idx, mask=mask, other=0.0)
        pos = tl.load(pos_ptr + n)
        angle = (pos * omega).to(tl.float32)
        s = tl.sin(angle)
        c = tl.cos(angle)
        out = x.to(tl.float32) * c + y.to(tl.float32) * s

        out_offsets = b * stride_out_b + h * stride_out_h + n * stride_out_n + cols * stride_out_d
        tl.store(out_ptr + out_offsets, out, mask=mask)


def can_use_triton_rope_rotate(x, pos):
    if not TRITON_AVAILABLE or not x.is_cuda or x.ndim != 4 or (x.size(-1) % 2) != 0:
        return False
    if x.dtype not in (torch.float16, torch.float32):
        return False
    if pos.ndim == 1 and pos.numel() == x.size(-2):
        return True
    if pos.ndim == 3 and pos.shape[0] == 1 and pos.shape[1] == 1 and pos.shape[2] == x.size(-2):
        return True
    return False


def triton_rotate_queries_or_keys(x, pos, omega, block_d=64):
    if pos.ndim != 1:
        pos = pos.reshape(-1)
    if pos.dtype != x.dtype:
        pos = pos.to(x.dtype)

    out = torch.empty_like(x)
    B, H, N, D = x.shape
    grid = (B * H * N, triton.cdiv(D, block_d))
    rope_rotate_kernel[grid](
        x,
        pos,
        omega,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        H,
        N,
        D,
        D // 2,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out
