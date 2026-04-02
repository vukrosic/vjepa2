"""Parity test for fused_rotate_queries_or_keys."""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.models.utils.kernels.fused_rotate_queries_or_keys import SHAPES, baseline_fn, can_use_kernel, kernel_fn


def test_forward_parity_cpu():
    shape = SHAPES["small"]
    x = torch.randn(*shape["x"], dtype=torch.float32)
    pos = torch.arange(shape["pos"][0], dtype=torch.float32)
    expected = baseline_fn(x, pos)
    actual = kernel_fn(x, pos)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape_name", ["small", "vit_l"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity_cuda(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    pos = torch.arange(shape["pos"][0], dtype=dtype, device="cuda")
    expected = baseline_fn(x, pos)
    actual = kernel_fn(x, pos)
    atol = 5e-3 if dtype == torch.float16 else 1e-4
    rtol = 5e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_accepts_broadcast_position_shape():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES["small"]
    x = torch.randn(*shape["x"], dtype=torch.float16, device="cuda")
    pos = torch.arange(shape["pos"][0], dtype=torch.float16, device="cuda").view(1, 1, -1)
    assert can_use_kernel(x, pos)
    expected = baseline_fn(x, pos)
    actual = kernel_fn(x, pos)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
