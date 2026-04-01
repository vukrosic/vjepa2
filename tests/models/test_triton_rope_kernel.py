# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from src.models.utils.modules import _INV_FREQ_CACHE, rotate_queries_or_keys
from src.models.utils.triton_kernels import TRITON_AVAILABLE, triton_rotate_queries_or_keys


def _get_omega(x):
    D = x.size(-1)
    key = (x.device.type, x.device.index, x.dtype, D)
    omega = _INV_FREQ_CACHE.get(key)
    if omega is None:
        omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
        omega /= D / 2.0
        omega = 1.0 / 10000**omega
        _INV_FREQ_CACHE[key] = omega
    return omega


def _baseline_rotate_queries_or_keys(x, pos):
    _, _, _, D = x.size()
    if pos.dtype != x.dtype:
        pos = pos.to(dtype=x.dtype)
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = pos.unsqueeze(-1) * omega
    emb_sin = freq.sin()
    emb_cos = freq.cos()
    emb_sin = emb_sin.squeeze(-1)
    emb_cos = emb_cos.squeeze(-1)
    emb_sin = torch.cat([emb_sin, emb_sin], dim=-1)
    emb_cos = torch.cat([emb_cos, emb_cos], dim=-1)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="This test requires CUDA + Triton")
def test_triton_rope_rotate_matches_pytorch_forward():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 128, 32, device="cuda", dtype=torch.float16)
    pos = torch.arange(128, device="cuda", dtype=torch.float16)
    omega = _get_omega(x)

    with torch.no_grad():
        reference = _baseline_rotate_queries_or_keys(x, pos)
        actual = triton_rotate_queries_or_keys(x, pos, omega)

    torch.testing.assert_close(actual, reference, atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="This test requires CUDA + Triton")
def test_live_rotate_queries_or_keys_matches_pytorch_backward():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 128, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    pos = torch.arange(128, device="cuda", dtype=torch.float16)

    reference = _baseline_rotate_queries_or_keys(x, pos)
    reference_loss = reference.square().mean()
    reference_loss.backward()
    reference_grad = x.grad.detach().clone()

    x.grad.zero_()
    actual = rotate_queries_or_keys(x, pos)
    actual_loss = actual.square().mean()
    actual_loss.backward()
    actual_grad = x.grad.detach().clone()

    torch.testing.assert_close(actual_grad, reference_grad, atol=3e-3, rtol=3e-3)
