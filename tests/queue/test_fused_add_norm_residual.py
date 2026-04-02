"""Parity test for fused_add_norm_residual kernel."""
import torch
import pytest

from src.models.utils.kernels.fused_add_norm_residual import can_use_kernel, kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 0.0
ATOL_FP32 = 0.0


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    D = shape["D"]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    b1 = torch.randn(shape["b1"], dtype=dtype, device="cuda")
    b2 = torch.randn(shape["b2"], dtype=dtype, device="cuda")
    weight = torch.randn(D, dtype=dtype, device="cuda")
    bias = torch.randn(D, dtype=dtype, device="cuda")
    expected = baseline_fn(x, b1, b2, weight, bias)
    actual = kernel_fn(x, b1, b2, weight, bias)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_backward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    D = shape["D"]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda", requires_grad=True)
    b1 = torch.randn(shape["b1"], dtype=dtype, device="cuda", requires_grad=True)
    b2 = torch.randn(shape["b2"], dtype=dtype, device="cuda", requires_grad=True)
    weight = torch.randn(D, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn(D, dtype=dtype, device="cuda", requires_grad=True)
    x_ker = x.detach().clone().requires_grad_(True)
    b1_ker = b1.detach().clone().requires_grad_(True)
    b2_ker = b2.detach().clone().requires_grad_(True)
    w_ker = weight.detach().clone().requires_grad_(True)
    bias_ker = bias.detach().clone().requires_grad_(True)
    y_ker = kernel_fn(x_ker, b1_ker, b2_ker, w_ker, bias_ker)
    y_bas = baseline_fn(x, b1, b2, weight, bias)
    y_ker.sum().backward()
    y_bas.sum().backward()
    atol = ATOL_FP16 if dtype == torch.float16 else 1e-6
    torch.testing.assert_close(x_ker.grad, x.grad, atol=atol, rtol=0)
    torch.testing.assert_close(b1_ker.grad, b1.grad, atol=atol, rtol=0)
    torch.testing.assert_close(b2_ker.grad, b2.grad, atol=atol, rtol=0)
    torch.testing.assert_close(w_ker.grad, weight.grad, atol=atol, rtol=0)
    torch.testing.assert_close(bias_ker.grad, bias.grad, atol=atol, rtol=0)


def test_can_use_kernel_strict_fallback():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES["small"]
    D = shape["D"]
    x = torch.randn(shape["x"], device="cuda")
    b1 = torch.randn(shape["b1"], device="cuda")
    b2 = torch.randn(shape["b2"], device="cuda")
    weight = torch.randn(D, device="cuda")
    bias = torch.randn(D, device="cuda")

    x_nc = x.transpose(1, 2).contiguous().transpose(1, 2)
    assert not can_use_kernel(x_nc, b1, b2, weight, bias)
    torch.testing.assert_close(
        kernel_fn(x_nc, b1, b2, weight, bias),
        baseline_fn(x_nc, b1, b2, weight, bias),
        atol=0.0,
        rtol=0,
    )
