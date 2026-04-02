"""Parity test for sorted target position helper."""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.models.utils.kernels.fused_sorted_target_positions import SHAPES, baseline_fn, can_use_kernel, kernel_fn


def make_masks(shape, device):
    masks_x = torch.sort(
        torch.randint(0, shape["max_index"], shape["masks_x"], device=device, dtype=torch.int64),
        dim=1,
    )[0]
    masks_y = torch.sort(
        torch.randint(0, shape["max_index"], shape["masks_y"], device=device, dtype=torch.int64),
        dim=1,
    )[0]
    return masks_x, masks_y


def test_forward_parity_cpu():
    masks_x, masks_y = make_masks(SHAPES["small"], "cpu")
    expected = baseline_fn(masks_x, masks_y)
    actual = kernel_fn(masks_x, masks_y)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity_cuda(shape_name):
    masks_x, masks_y = make_masks(SHAPES[shape_name], "cuda")
    expected = baseline_fn(masks_x, masks_y)
    actual = kernel_fn(masks_x, masks_y)
    assert can_use_kernel(masks_x, masks_y)
    torch.testing.assert_close(actual, expected)


def test_fallback_on_rank_mismatch():
    masks_x = torch.tensor([0, 2, 4], dtype=torch.int64)
    masks_y = torch.tensor([1, 3], dtype=torch.int64)
    actual = kernel_fn(masks_x, masks_y)
    expected = baseline_fn(masks_x.unsqueeze(0), masks_y.unsqueeze(0)).squeeze(0)
    torch.testing.assert_close(actual, expected)
