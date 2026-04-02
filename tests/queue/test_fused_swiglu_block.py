"""Parity test for fused SiGLU Block kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_swiglu_block import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    w1 = torch.randn(shape["N"], shape["M"], dtype=dtype, device="cuda")
    b1 = torch.randn(shape["N"], dtype=dtype, device="cuda")
    w2 = torch.randn(shape["N"], shape["M"], dtype=dtype, device="cuda")
    b2 = torch.randn(shape["N"], dtype=dtype, device="cuda")
    expected = baseline_fn(x, w1, b1, w2, b2)
    actual = kernel_fn(x, w1, b1, w2, b2)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
