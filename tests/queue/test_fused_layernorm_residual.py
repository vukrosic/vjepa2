"""Parity test for fused_layernorm_residual kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_layernorm_residual import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    D = shape["D"]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    residual = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    weight = torch.ones(D, dtype=dtype, device="cuda")
    bias = torch.zeros(D, dtype=dtype, device="cuda")
    expected = baseline_fn(x, residual, weight, bias)
    actual = kernel_fn(x, residual, weight, bias)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    D = shape["D"]
    xa = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    ra = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    weight_a = torch.ones(D, dtype=torch.float32, device="cuda", requires_grad=True)
    bias_a = torch.zeros(D, dtype=torch.float32, device="cuda", requires_grad=True)

    xb = xa.detach().clone().requires_grad_(True)
    rb = ra.detach().clone().requires_grad_(True)
    weight_b = weight_a.detach().clone().requires_grad_(True)
    bias_b = bias_a.detach().clone().requires_grad_(True)

    out_base = baseline_fn(xa, ra, weight_a, bias_a)
    out_kern = kernel_fn(xb, rb, weight_b, bias_b)

    grad = torch.randn_like(out_base)
    out_base.backward(grad)
    out_kern.backward(grad)

    torch.testing.assert_close(xb.grad, xa.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(rb.grad, ra.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(weight_b.grad, weight_a.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(bias_b.grad, bias_a.grad, atol=1e-4, rtol=1e-4)
