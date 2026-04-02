"""Parity test for repeat_interleave_batch kernel family."""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.models.utils.kernels.fused_repeat_interleave_batch import SHAPES, baseline_fn, can_use_kernel, kernel_fn


def test_forward_parity_cpu():
    shape = SHAPES["small"]
    x = torch.arange(shape["x"][0] * shape["x"][1], dtype=torch.float32).reshape(*shape["x"])
    expected = baseline_fn(x, shape["B"], shape["repeat"])
    actual = kernel_fn(x, shape["B"], shape["repeat"])
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity_cuda(shape_name):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], device="cuda", dtype=torch.float16)
    expected = baseline_fn(x, shape["B"], shape["repeat"])
    actual = kernel_fn(x, shape["B"], shape["repeat"])
    assert can_use_kernel(x, shape["B"], shape["repeat"])
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_repeat_1_returns_input():
    x = torch.randn(6, 4)
    actual = kernel_fn(x, 2, 1)
    torch.testing.assert_close(actual, x)
