# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from src.models.utils.modules import ACRoPEAttention, Attention, RoPEAttention


def test_attention_eval_sdpa_is_deterministic_cpu():
    torch.manual_seed(0)
    module = Attention(dim=64, num_heads=4, qkv_bias=True, proj_drop=0.5, use_sdpa=True).eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        first = module(x)
        second = module(x)
    torch.testing.assert_close(first, second)


def test_rope_attention_eval_sdpa_is_deterministic_cpu():
    torch.manual_seed(0)
    module = RoPEAttention(dim=64, num_heads=4, qkv_bias=True, proj_drop=0.5, use_sdpa=True, grid_size=2).eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        first = module(x, T=2, H_patches=2, W_patches=2)
        second = module(x, T=2, H_patches=2, W_patches=2)
    torch.testing.assert_close(first, second)


def test_ac_rope_attention_eval_sdpa_is_deterministic_cpu():
    torch.manual_seed(0)
    module = ACRoPEAttention(dim=64, num_heads=4, qkv_bias=True, proj_drop=0.5, use_sdpa=True, grid_size=2).eval()
    x = torch.randn(1, 2 * (2 + 4), 64)
    with torch.no_grad():
        first = module(x, T=2, H=2, W=2, action_tokens=2)
        second = module(x, T=2, H=2, W=2, action_tokens=2)
    torch.testing.assert_close(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires CUDA")
def test_rope_attention_half_cuda_unmasked_runs():
    module = RoPEAttention(dim=64, num_heads=4, qkv_bias=True, proj_drop=0.0, use_sdpa=True, grid_size=2).cuda().half().eval()
    x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        out = module(x, T=2, H_patches=2, W_patches=2)
    assert out.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires CUDA")
def test_ac_rope_attention_half_cuda_unmasked_runs():
    module = ACRoPEAttention(dim=64, num_heads=4, qkv_bias=True, proj_drop=0.0, use_sdpa=True, grid_size=2).cuda().half().eval()
    x = torch.randn(1, 2 * (2 + 4), 64, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        out = module(x, T=2, H=2, W=2, action_tokens=2)
    assert out.dtype == torch.float16
