"""Parity test for fused layer_norm kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_layer_norm import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-1
ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    weight = torch.randn(*shape["weight"], dtype=dtype, device="cuda")
    bias = torch.randn(*shape["bias"], dtype=dtype, device="cuda")
    expected = baseline_fn(x, weight, bias)
    actual = kernel_fn(x, weight, bias)
    torch.testing.assert_close(actual, expected, atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32, rtol=1e-3)
