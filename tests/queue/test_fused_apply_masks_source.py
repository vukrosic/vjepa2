"""Parity test for fused_apply_masks_source."""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.models.utils.kernels.fused_apply_masks_source import SHAPES, baseline_fn, kernel_fn


def _make_masks(shape, device):
    if shape["kind"] == "1d":
        return [
            torch.randint(0, shape["x"][1], (shape["mask_len"],), device=device, dtype=torch.long)
            for _ in range(shape["mask_count"])
        ]
    bsz, k = shape["mask_shape"]
    return [
        torch.randint(0, shape["x"][1], (bsz, k), device=device, dtype=torch.long)
        for _ in range(shape["mask_count"])
    ]


@pytest.mark.parametrize("shape_name", ["multi1d", "multi2d"])
def test_forward_parity_cpu(shape_name):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=torch.float32)
    masks = _make_masks(shape, "cpu")
    expected = baseline_fn(x, masks)
    actual = kernel_fn(x, masks)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity_cuda(shape_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=torch.float16, device="cuda")
    masks = _make_masks(shape, "cuda")
    expected = baseline_fn(x, masks)
    actual = kernel_fn(x, masks)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0)


def test_concat_false_parity_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES["multi2d"]
    x = torch.randn(*shape["x"], dtype=torch.float16, device="cuda")
    masks = _make_masks(shape, "cuda")
    expected = baseline_fn(x, masks, concat=False)
    actual = kernel_fn(x, masks, concat=False)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0.0, rtol=0)
