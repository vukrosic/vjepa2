"""Parity test for fused Subtract Mean + Scale kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_subtract_mean_scale import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    mean = torch.randn(shape["C"], dtype=dtype, device="cuda")
    gamma = torch.randn(shape["C"], dtype=dtype, device="cuda")
    expected = baseline_fn(x, mean, gamma)
    actual = kernel_fn(x, mean, gamma)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    xa = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    ma = torch.randn(shape["C"], dtype=torch.float32, device="cuda", requires_grad=True)
    ga = torch.randn(shape["C"], dtype=torch.float32, device="cuda", requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    mb = ma.detach().clone().requires_grad_(True)
    gb = ga.detach().clone().requires_grad_(True)

    out_base = baseline_fn(xa, ma, ga)
    out_kern = kernel_fn(xb, mb, gb)

    grad = torch.randn_like(out_base)
    out_base.backward(grad)
    out_kern.backward(grad)

    torch.testing.assert_close(xb.grad, xa.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(mb.grad, ma.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(gb.grad, ga.grad, atol=1e-4, rtol=1e-4)
