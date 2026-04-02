"""Parity test for fused Add Bias + GELU kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_add_bias_gelu import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    bias = torch.randn(*shape["bias"], dtype=dtype, device="cuda")
    expected = baseline_fn(x, bias)
    actual = kernel_fn(x, bias)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    xa = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    ba = torch.randn(*shape["bias"], dtype=torch.float32, device="cuda", requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    bb = ba.detach().clone().requires_grad_(True)

    out_base = baseline_fn(xa, ba)
    out_kern = kernel_fn(xb, bb)

    grad = torch.randn_like(out_base)
    out_base.backward(grad)
    out_kern.backward(grad)

    torch.testing.assert_close(xb.grad, xa.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(bb.grad, ba.grad, atol=1e-4, rtol=1e-4)
