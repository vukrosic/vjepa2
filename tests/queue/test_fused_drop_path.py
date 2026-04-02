"""Parity test for fused drop_path kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_drop_path import kernel_fn, SHAPES

ATOL_FP16 = 1e-1


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=torch.float32, device="cuda")
    expected = x  # inference mode: identity
    actual = kernel_fn(x, drop_prob=0.0, training=False)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
