"""Parity test for fused binary_cross_entropy kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_binary_cross_entropy import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-1
ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    pred = torch.rand(*shape["pred"], dtype=dtype, device="cuda")
    target = torch.rand(*shape["target"], dtype=dtype, device="cuda")
    expected = baseline_fn(pred, target)
    actual = kernel_fn(pred, target)
    torch.testing.assert_close(actual, expected, atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32, rtol=1e-3)
