"""Parity test for fused_rotate_query_key_pair."""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.models.utils.kernels.fused_rotate_query_key_pair import SHAPES, baseline_fn, can_use_kernel, kernel_fn


@pytest.mark.parametrize("shape_name", ["small"])
def test_forward_parity_cpu(shape_name):
    shape = SHAPES[shape_name]
    q = torch.randn(*shape["q"], dtype=torch.float32)
    k = torch.randn(*shape["k"], dtype=torch.float32)
    pos = torch.arange(shape["pos"][-1], dtype=torch.float32).view(*shape["pos"])
    expected_q, expected_k = baseline_fn(q, k, pos)
    actual_q, actual_k = kernel_fn(q, k, pos)
    torch.testing.assert_close(actual_q, expected_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape_name", ["small", "vit_l"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity_cuda(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    q = torch.randn(*shape["q"], dtype=dtype, device="cuda")
    k = torch.randn(*shape["k"], dtype=dtype, device="cuda")
    pos = torch.arange(shape["pos"][-1], dtype=dtype, device="cuda").view(*shape["pos"])
    expected_q, expected_k = baseline_fn(q, k, pos)
    actual_q, actual_k = kernel_fn(q, k, pos)
    atol = 5e-3 if dtype == torch.float16 else 1e-4
    rtol = 5e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(actual_q, expected_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_k, expected_k, atol=atol, rtol=rtol)


def test_supports_1d_positions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES["small"]
    q = torch.randn(*shape["q"], dtype=torch.float16, device="cuda")
    k = torch.randn(*shape["k"], dtype=torch.float16, device="cuda")
    pos = torch.arange(shape["pos"][-1], dtype=torch.float16, device="cuda")
    assert can_use_kernel(q, k, pos)
    expected_q, expected_k = baseline_fn(q, k, pos)
    actual_q, actual_k = kernel_fn(q, k, pos)
    torch.testing.assert_close(actual_q, expected_q, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(actual_k, expected_k, atol=5e-3, rtol=5e-3)
