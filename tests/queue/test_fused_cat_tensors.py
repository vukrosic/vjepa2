"""Parity test for fused cat_tensors kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_cat_tensors import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    a = torch.randn(*shape["a"], dtype=dtype, device="cuda")
    b = torch.randn(*shape["b"], dtype=dtype, device="cuda")
    expected = baseline_fn(a, b, dim=0)
    actual = kernel_fn(a, b, dim=0)
    torch.testing.assert_close(actual, expected, atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32, rtol=1e-4)
