"""Parity test for fused_layernorm_rmsnorm kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_layernorm_rmsnorm import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    ln_D = shape["LN_D"]
    rms_D = shape["RMS_D"]
    D = ln_D + rms_D
    ln_weight = torch.ones(ln_D, dtype=dtype, device="cuda")
    ln_bias = torch.zeros(ln_D, dtype=dtype, device="cuda")
    rms_weight = torch.ones(rms_D, dtype=dtype, device="cuda")
    expected = baseline_fn(x, ln_weight, ln_bias, rms_weight, ln_D)
    actual = kernel_fn(x, ln_weight, ln_bias, rms_weight, ln_D)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    ln_D = shape["LN_D"]
    rms_D = shape["RMS_D"]
    ln_weight = torch.ones(ln_D, dtype=torch.float32, device="cuda")
    ln_bias = torch.zeros(ln_D, dtype=torch.float32, device="cuda")
    rms_weight = torch.ones(rms_D, dtype=torch.float32, device="cuda")
    x_ker = x.detach().clone().requires_grad_(True)
    out1 = baseline_fn(x, ln_weight, ln_bias, rms_weight, ln_D)
    out2 = kernel_fn(x_ker, ln_weight, ln_bias, rms_weight, ln_D)
    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    torch.testing.assert_close(x_ker.grad, x.grad, atol=1e-3, rtol=1e-3)
