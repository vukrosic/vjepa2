"""Parity test for fused Percentile Clip kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_percentile_clip import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    low = torch.randn(shape["C"], dtype=dtype, device="cuda")
    high = torch.randn(shape["C"], dtype=dtype, device="cuda")
    # Ensure low < high
    low = torch.minimum(low, high) - 0.1
    high = torch.maximum(low, high) + 0.1
    expected = baseline_fn(x, low, high)
    actual = kernel_fn(x, low, high)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
