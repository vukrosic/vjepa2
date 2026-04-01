# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import torch

from src.models.utils.modules import ACRoPEAttention, rotate_queries_or_keys


def reference_ac_rope_forward(module, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
    B, N, C = x.size()

    if mask is not None:
        mask = mask.unsqueeze(1).repeat(1, module.num_heads, 1)
        d_mask, h_mask, w_mask = module.separate_positions(mask, H, W)
        h_mask *= module.grid_size / H
        w_mask *= module.grid_size / W
    else:
        pos_mask = torch.arange(int(T * H * W), device=x.device)
        d_mask, h_mask, w_mask = module.separate_positions(pos_mask, H, W)
        h_mask *= module.grid_size / H
        w_mask *= module.grid_size / W

    if action_tokens > 0:
        x = x.view(B, -1, action_tokens + H * W, C)
        action_q, action_k, action_v = [], [], []
        for i in range(action_tokens):
            a = x[:, :, i : i + 1, :].flatten(1, 2)
            qkv = module.qkv(a).unflatten(-1, (3, module.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            t_pos = torch.arange(T, device=x.device)
            qd = rotate_queries_or_keys(q[..., : module.d_dim], pos=t_pos)
            kd = rotate_queries_or_keys(k[..., : module.d_dim], pos=t_pos)
            qr = q[..., module.d_dim :]
            kr = k[..., module.d_dim :]
            action_q += [torch.cat([qd, qr], dim=-1).view(B, module.num_heads, T, 1, -1)]
            action_k += [torch.cat([kd, kr], dim=-1).view(B, module.num_heads, T, 1, -1)]
            action_v += [v.view(B, module.num_heads, T, 1, -1)]

        action_q = torch.cat(action_q, dim=3).flatten(2, 3)
        action_k = torch.cat(action_k, dim=3).flatten(2, 3)
        action_v = torch.cat(action_v, dim=3).flatten(2, 3)
        x = x[:, :, action_tokens:, :].flatten(1, 2)

    qkv = module.qkv(x).unflatten(-1, (3, module.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    s = 0
    qd = rotate_queries_or_keys(q[..., s : s + module.d_dim], pos=d_mask)
    kd = rotate_queries_or_keys(k[..., s : s + module.d_dim], pos=d_mask)
    s += module.d_dim
    qh = rotate_queries_or_keys(q[..., s : s + module.h_dim], pos=h_mask)
    kh = rotate_queries_or_keys(k[..., s : s + module.h_dim], pos=h_mask)
    s += module.h_dim
    qw = rotate_queries_or_keys(q[..., s : s + module.w_dim], pos=w_mask)
    kw = rotate_queries_or_keys(k[..., s : s + module.w_dim], pos=w_mask)
    s += module.w_dim

    if s < module.head_dim:
        qr = q[..., s:]
        kr = k[..., s:]
        q = torch.cat([qd, qh, qw, qr], dim=-1)
        k = torch.cat([kd, kh, kw, kr], dim=-1)
    else:
        q = torch.cat([qd, qh, qw], dim=-1)
        k = torch.cat([kd, kh, kw], dim=-1)

    if action_tokens > 0:
        q = torch.cat([action_q.view(B, module.num_heads, T, action_tokens, -1), q.view(B, module.num_heads, T, H * W, -1)], dim=3).flatten(2, 3)
        k = torch.cat([action_k.view(B, module.num_heads, T, action_tokens, -1), k.view(B, module.num_heads, T, H * W, -1)], dim=3).flatten(2, 3)
        v = torch.cat([action_v.view(B, module.num_heads, T, action_tokens, -1), v.view(B, module.num_heads, T, H * W, -1)], dim=3).flatten(2, 3)

    attn = (q @ k.transpose(-2, -1)) * module.scale
    attn = attn.softmax(dim=-1)
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = module.proj(x)
    x = module.proj_drop(x)
    return x


class TestACRoPEAttention(unittest.TestCase):
    def test_cpu_matches_reference_action_path(self):
        torch.manual_seed(0)
        module = ACRoPEAttention(dim=64, num_heads=4, qkv_bias=True, use_sdpa=False, grid_size=2).eval()
        x = torch.randn(1, 2 * (2 + 4), 64)
        with torch.no_grad():
            actual = module(x, T=2, H=2, W=2, action_tokens=2)
            expected = reference_ac_rope_forward(module, x, T=2, H=2, W=2, action_tokens=2)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires CUDA")
    def test_cuda_matches_reference_action_path(self):
        torch.manual_seed(0)
        module = ACRoPEAttention(dim=64, num_heads=4, qkv_bias=True, use_sdpa=False, grid_size=2).cuda().eval()
        x = torch.randn(1, 2 * (2 + 4), 64, device="cuda")
        with torch.no_grad():
            actual = module(x, T=2, H=2, W=2, action_tokens=2)
            expected = reference_ac_rope_forward(module, x, T=2, H=2, W=2, action_tokens=2)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
