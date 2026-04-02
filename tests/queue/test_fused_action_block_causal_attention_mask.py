"""Parity test for build_action_block_causal_attention_mask."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.models.utils.kernels.fused_action_block_causal_attention_mask import (
    SHAPES,
    baseline_fn,
    can_use_kernel,
    kernel_fn,
)


def test_parity_matches_baseline():
    for shape in SHAPES.values():
        expected = baseline_fn(shape["T"], shape["H"], shape["W"], shape["add_tokens"])
        actual = kernel_fn(shape["T"], shape["H"], shape["W"], shape["add_tokens"])
        torch.testing.assert_close(actual, expected)


def test_returns_fresh_tensor():
    shape = SHAPES["small"]
    mask_a = kernel_fn(shape["T"], shape["H"], shape["W"], shape["add_tokens"])
    mask_b = kernel_fn(shape["T"], shape["H"], shape["W"], shape["add_tokens"])
    assert can_use_kernel(shape["T"], shape["H"], shape["W"], shape["add_tokens"])
    mask_a[0, 0] = False
    assert mask_b[0, 0].item() is True


def test_falls_back_on_bad_inputs():
    expected = baseline_fn(2, 2, 2, 1)
    actual = kernel_fn(2.0, 2, 2, 1)
    torch.testing.assert_close(actual, expected)
