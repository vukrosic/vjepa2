"""Parity test for fused_dropout_residual_norm kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_dropout_residual_norm import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    residual = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    D = shape["D"]
    weight = torch.ones(D, dtype=dtype, device="cuda")
    bias = torch.zeros(D, dtype=dtype, device="cuda")
    # Training=True uses dropout; test with training=False for deterministic test
    expected = baseline_fn(x, residual, weight, bias, p=0.0, training=False)
    actual = kernel_fn(x, residual, weight, bias, p=0.0, training=False, seed=42)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    residual = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    D = shape["D"]
    weight = torch.ones(D, dtype=torch.float32, device="cuda")
    bias = torch.zeros(D, dtype=torch.float32, device="cuda")
    x_ker = x.detach().clone().requires_grad_(True)
    res_ker = residual.detach().clone().requires_grad_(True)
    out1 = baseline_fn(x, residual, weight, bias, p=0.0, training=False)
    out2 = kernel_fn(x_ker, res_ker, weight, bias, p=0.0, training=False, seed=42)
    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    torch.testing.assert_close(x_ker.grad, x.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(res_ker.grad, residual.grad, atol=1e-2, rtol=1e-2)
