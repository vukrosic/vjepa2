# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from src.masks.utils import apply_masks
from src.models.utils.modules import build_action_block_causal_attention_mask, rotate_queries_or_keys
from src.utils.tensors import repeat_interleave_batch


def _baseline_action_block_causal_attention_mask(T, H, W, add_tokens=1):
    tokens_per_step = add_tokens + (H * W)
    total_tokens = T * tokens_per_step
    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool)
    block = torch.ones(tokens_per_step, tokens_per_step, dtype=torch.bool)
    for t1 in range(T):
        for t2 in range(max(0, t1 - T + 1), t1 + 1):
            mask[t1 * tokens_per_step : (t1 + 1) * tokens_per_step, t2 * tokens_per_step : (t2 + 1) * tokens_per_step] = block
    return mask


def _baseline_repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    return torch.cat([torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)], dim=0)


def _baseline_apply_masks(x, masks, concat=True):
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def _baseline_rotate_queries_or_keys(x, pos):
    _, _, _, D = x.size()
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freq.sin()
    emb_cos = freq.cos()
    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


def test_action_block_causal_attention_mask_matches_baseline():
    baseline = _baseline_action_block_causal_attention_mask(T=4, H=3, W=2, add_tokens=2)
    optimized = build_action_block_causal_attention_mask(T=4, H=3, W=2, add_tokens=2)
    torch.testing.assert_close(optimized, baseline)


def test_action_block_causal_attention_mask_returns_fresh_tensor():
    mask_a = build_action_block_causal_attention_mask(T=2, H=2, W=2, add_tokens=2)
    mask_b = build_action_block_causal_attention_mask(T=2, H=2, W=2, add_tokens=2)
    mask_a[0, 0] = False
    assert mask_b[0, 0].item() is True


def test_repeat_interleave_batch_matches_baseline_cpu():
    x = torch.randn(12, 5)
    baseline = _baseline_repeat_interleave_batch(x, B=3, repeat=4)
    optimized = repeat_interleave_batch(x, B=3, repeat=4)
    torch.testing.assert_close(optimized, baseline)


def test_repeat_interleave_batch_matches_baseline_cuda():
    if not torch.cuda.is_available():
        return
    x = torch.randn(12, 5, device="cuda")
    baseline = _baseline_repeat_interleave_batch(x, B=3, repeat=4)
    optimized = repeat_interleave_batch(x, B=3, repeat=4)
    torch.testing.assert_close(optimized, baseline)


def test_apply_masks_matches_baseline_cpu():
    x = torch.randn(2, 8, 4)
    masks = [torch.tensor([[0, 2, 4], [1, 3, 5]]), torch.tensor([[6, 7, 1], [0, 2, 4]])]
    baseline = _baseline_apply_masks(x, masks)
    optimized = apply_masks(x, masks)
    torch.testing.assert_close(optimized, baseline)


def test_apply_masks_matches_baseline_cpu_single_mask_rows():
    x = torch.randn(4, 8, 4)
    masks = [torch.tensor([0, 2, 4]), torch.tensor([1, 3, 5]), torch.tensor([2, 4, 6]), torch.tensor([3, 5, 7])]
    baseline = _baseline_apply_masks(x, masks)
    optimized = apply_masks(x, masks)
    torch.testing.assert_close(optimized, baseline)


def test_apply_masks_matches_baseline_cuda():
    if not torch.cuda.is_available():
        return
    x = torch.randn(2, 8, 4, device="cuda")
    masks = [
        torch.tensor([[0, 2, 4], [1, 3, 5]], device="cuda"),
        torch.tensor([[6, 7, 1], [0, 2, 4]], device="cuda"),
    ]
    baseline = _baseline_apply_masks(x, masks)
    optimized = apply_masks(x, masks)
    torch.testing.assert_close(optimized, baseline)


def test_rotate_queries_or_keys_matches_baseline_cpu():
    x = torch.randn(2, 3, 8, 12)
    pos = torch.arange(8).view(1, 1, 8)
    baseline = _baseline_rotate_queries_or_keys(x, pos)
    optimized = rotate_queries_or_keys(x, pos)
    torch.testing.assert_close(optimized, baseline)


def test_rotate_queries_or_keys_matches_baseline_cuda():
    if not torch.cuda.is_available():
        return
    x = torch.randn(2, 3, 8, 12, device="cuda")
    pos = torch.arange(8, device="cuda").view(1, 1, 8)
    baseline = _baseline_rotate_queries_or_keys(x, pos)
    optimized = rotate_queries_or_keys(x, pos)
    torch.testing.assert_close(optimized, baseline)


def test_rotate_queries_or_keys_matches_baseline_cuda_1d_pos():
    if not torch.cuda.is_available():
        return
    x = torch.randn(2, 3, 8, 12, device="cuda", dtype=torch.float16)
    pos = torch.arange(8, device="cuda")
    baseline = _baseline_rotate_queries_or_keys(x, pos)
    optimized = rotate_queries_or_keys(x, pos)
    torch.testing.assert_close(optimized, baseline, atol=3e-3, rtol=3e-3)
