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


    @triton.jit
    def rope_rotate_pair_kernel(
        q_ptr,
        k_ptr,
        pos_ptr,
        omega_ptr,
        out_q_ptr,
        out_k_ptr,
        stride_q_b,
        stride_q_h,
        stride_q_n,
        stride_q_d,
        stride_k_b,
        stride_k_h,
        stride_k_n,
        stride_k_d,
        stride_out_q_b,
        stride_out_q_h,
        stride_out_q_n,
        stride_out_q_d,
        stride_out_k_b,
        stride_out_k_h,
        stride_out_k_n,
        stride_out_k_d,
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

        q_offsets = b * stride_q_b + h * stride_q_h + n * stride_q_n + cols * stride_q_d
        k_offsets = b * stride_k_b + h * stride_k_h + n * stride_k_n + cols * stride_k_d
        pair_cols = cols ^ 1
        q_pair_offsets = b * stride_q_b + h * stride_q_h + n * stride_q_n + pair_cols * stride_q_d
        k_pair_offsets = b * stride_k_b + h * stride_k_h + n * stride_k_n + pair_cols * stride_k_d

        q = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)
        q_pair = tl.load(q_ptr + q_pair_offsets, mask=mask, other=0.0)
        q_rot = tl.where((cols & 1) == 0, -q_pair, q_pair)

        k = tl.load(k_ptr + k_offsets, mask=mask, other=0.0)
        k_pair = tl.load(k_ptr + k_pair_offsets, mask=mask, other=0.0)
        k_rot = tl.where((cols & 1) == 0, -k_pair, k_pair)

        omega_idx = cols % HALF_D
        omega = tl.load(omega_ptr + omega_idx, mask=mask, other=0.0)
        pos = tl.load(pos_ptr + n)
        angle = (pos * omega).to(tl.float32)
        s = tl.sin(angle)
        c = tl.cos(angle)

        out_q = q.to(tl.float32) * c + q_rot.to(tl.float32) * s
        out_k = k.to(tl.float32) * c + k_rot.to(tl.float32) * s

        out_q_offsets = b * stride_out_q_b + h * stride_out_q_h + n * stride_out_q_n + cols * stride_out_q_d
        out_k_offsets = b * stride_out_k_b + h * stride_out_k_h + n * stride_out_k_n + cols * stride_out_k_d
        tl.store(out_q_ptr + out_q_offsets, out_q, mask=mask)
        tl.store(out_k_ptr + out_k_offsets, out_k, mask=mask)


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
        num_warps=2,
    )
    return out


def triton_rotate_query_key_pair(q, k, pos, omega, block_d=128):
    if pos.ndim != 1:
        pos = pos.reshape(-1)
    if pos.dtype != q.dtype:
        pos = pos.to(q.dtype)

    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)
    B, H, N, D = q.shape
    grid = (B * H * N, triton.cdiv(D, block_d))
    rope_rotate_pair_kernel[grid](
        q,
        k,
        pos,
        omega,
        out_q,
        out_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        out_q.stride(0),
        out_q.stride(1),
        out_q.stride(2),
        out_q.stride(3),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        out_k.stride(3),
        H,
        N,
        D,
        D // 2,
        BLOCK_D=block_d,
        num_warps=2,
    )
    return out_q, out_k


def triton_rotate_query_key_pair_autograd(q, k, pos, omega):
    return _TritonRotateQueryKeyPairAutograd.apply(q, k, pos, omega)


def triton_rotate_queries_or_keys_autograd(x, pos, omega):
    return _TritonRotateQueriesOrKeysAutograd.apply(x, pos, omega)


class _TritonRotateQueriesOrKeysAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pos, omega):
        if pos.ndim != 1:
            pos = pos.reshape(-1)
        if pos.dtype != x.dtype:
            pos = pos.to(x.dtype)
        ctx.save_for_backward(pos, omega)
        return triton_rotate_queries_or_keys(x, pos, omega)

    @staticmethod
    def backward(ctx, grad_output):
        pos, omega = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x = triton_rotate_queries_or_keys(grad_output, -pos, omega)
        return grad_x, None, None


class _TritonRotateQueryKeyPairAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, pos, omega):
        if pos.ndim != 1:
            pos = pos.reshape(-1)
        if pos.dtype != q.dtype:
            pos = pos.to(q.dtype)
        ctx.save_for_backward(pos, omega)
        return triton_rotate_query_key_pair(q, k, pos, omega)

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        pos, omega = ctx.saved_tensors
        grad_q = grad_q.contiguous()
        grad_k = grad_k.contiguous()
        grad_q, grad_k = triton_rotate_query_key_pair(grad_q, grad_k, -pos, omega)
        return grad_q, grad_k, None, None
