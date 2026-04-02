"""Parity test for fused_swiglu kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_swiglu import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    w1 = torch.randn(shape["w1"], dtype=dtype, device="cuda")
    w2 = torch.randn(shape["w2"], dtype=dtype, device="cuda")
    expected = baseline_fn(x, w1, w2)
    actual = kernel_fn(x, w1, w2)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    w1 = torch.randn(shape["w1"], dtype=torch.float32, device="cuda", requires_grad=True)
    w2 = torch.randn(shape["w2"], dtype=torch.float32, device="cuda", requires_grad=True)
    x_ker = x.detach().clone().requires_grad_(True)
    w1_ker = w1.detach().clone().requires_grad_(True)
    w2_ker = w2.detach().clone().requires_grad_(True)
    y_ker = kernel_fn(x_ker, w1_ker, w2_ker)
    y_bas = baseline_fn(x, w1, w2)
    grad = torch.randn_like(y_ker)
    y_ker.backward(grad)
    y_bas.backward(grad)
    torch.testing.assert_close(x_ker.grad, x.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(w1_ker.grad, w1.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(w2_ker.grad, w2.grad, atol=1e-3, rtol=1e-3)
