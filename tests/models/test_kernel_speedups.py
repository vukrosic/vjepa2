# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import torch

from app.vjepa_2_1.models.utils.modules import rotate_queries_or_keys as app_rotate_queries_or_keys
from src.models.utils.modules import build_action_block_causal_attention_mask, rotate_queries_or_keys
from src.utils.tensors import repeat_interleave_batch


def baseline_build_action_block_causal_attention_mask(T, H, W, add_tokens=1):
    n_t = add_tokens + (H * W)
    n = T * n_t
    mask = torch.zeros(n, n).bool()
    mask_block = torch.ones(n_t, n_t).bool()
    local_window_time = T

    for t1 in range(T):
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            mask[t1 * n_t : (t1 + 1) * n_t, t2 * n_t : (t2 + 1) * n_t] = mask_block

    return mask


def baseline_rotate_queries_or_keys(x, pos):
    D = x.size(-1)
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = pos.unsqueeze(-1) * omega
    emb_sin = freq.sin().squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = freq.cos().squeeze(-1).repeat(1, 1, 1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


def baseline_app_rotate_queries_or_keys(x, pos, n_registers, has_cls_first):
    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = x.size(-2) - n_registers

    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., start_ctx:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

    omega = torch.arange(x_ctx.size(-1) // 2, dtype=x_ctx.dtype, device=x_ctx.device)
    omega /= x_ctx.size(-1) / 2.0
    omega = 1.0 / 10000**omega
    freq = pos.unsqueeze(-1) * omega
    emb_sin = freq.sin().repeat_interleave(2, dim=-1)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)
    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    out_ctx = (x_ctx * emb_cos) + (y * emb_sin)

    parts = []
    if n_cls:
        parts.append(x_cls)
    parts.append(out_ctx)
    if n_registers:
        parts.append(x_reg)
    return torch.cat(parts, dim=-2)


def baseline_repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    return torch.cat([torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)], dim=0)


class TestKernelParity(unittest.TestCase):
    def test_action_block_causal_attention_mask_matches_baseline(self):
        actual = build_action_block_causal_attention_mask(T=4, H=3, W=5, add_tokens=2)
        expected = baseline_build_action_block_causal_attention_mask(T=4, H=3, W=5, add_tokens=2)
        torch.testing.assert_close(actual, expected)

    def test_repeat_interleave_batch_matches_baseline(self):
        x = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        actual = repeat_interleave_batch(x, B=2, repeat=3)
        expected = baseline_repeat_interleave_batch(x, B=2, repeat=3)
        torch.testing.assert_close(actual, expected)

    def test_rotate_queries_or_keys_matches_baseline_cpu(self):
        x = torch.randn(2, 3, 7, 8)
        pos = torch.arange(7, dtype=x.dtype).view(1, 1, 7)
        actual = rotate_queries_or_keys(x, pos)
        expected = baseline_rotate_queries_or_keys(x, pos)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires CUDA")
    def test_rotate_queries_or_keys_matches_baseline_cuda(self):
        x = torch.randn(2, 4, 13, 32, device="cuda", dtype=torch.float16)
        pos = torch.arange(13, device="cuda", dtype=x.dtype).view(1, 1, 13)
        actual = rotate_queries_or_keys(x, pos)
        expected = baseline_rotate_queries_or_keys(x, pos)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires CUDA")
    def test_app_rotate_queries_or_keys_matches_baseline_cuda(self):
        x = torch.randn(2, 4, 17, 32, device="cuda", dtype=torch.float16)
        pos = torch.arange(12, device="cuda", dtype=x.dtype).view(1, 1, 12)
        actual = app_rotate_queries_or_keys(x, pos, n_registers=4, has_cls_first=True)
        expected = baseline_app_rotate_queries_or_keys(x, pos, n_registers=4, has_cls_first=True)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
