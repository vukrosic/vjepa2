"""Parity test for fused_norm_linear kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_norm_linear import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"][0], shape["x"][1], shape["D_in"], dtype=dtype, device="cuda")
    weight = torch.randn(shape["D_out"], shape["D_in"], dtype=dtype, device="cuda")
    bias = torch.randn(shape["D_out"], dtype=dtype, device="cuda")
    ln_weight = torch.ones(shape["D_in"], dtype=dtype, device="cuda")
    ln_bias = torch.zeros(shape["D_in"], dtype=dtype, device="cuda")
    expected = baseline_fn(x, weight, bias, ln_weight, ln_bias)
    actual = kernel_fn(x, weight, bias, ln_weight, ln_bias)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
