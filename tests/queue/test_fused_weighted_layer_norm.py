"""Parity test for fused Weighted LayerNorm kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_weighted_layer_norm import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    weight = torch.randn(shape["C"], dtype=dtype, device="cuda")
    bias = torch.randn(shape["C"], dtype=dtype, device="cuda")
    token_weight = torch.rand(shape["x"][0], shape["x"][1], 1, dtype=dtype, device="cuda") + 0.5
    expected = baseline_fn(x, weight, bias, token_weight)
    actual = kernel_fn(x, weight, bias, token_weight)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    xa = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    wa = torch.randn(shape["C"], dtype=torch.float32, device="cuda", requires_grad=True)
    ba = torch.randn(shape["C"], dtype=torch.float32, device="cuda", requires_grad=True)
    twa = (torch.rand(shape["x"][0], shape["x"][1], 1, dtype=torch.float32, device="cuda") + 0.5).requires_grad_()
    xb = xa.detach().clone().requires_grad_(True)
    wb = wa.detach().clone().requires_grad_(True)
    bb = ba.detach().clone().requires_grad_(True)
    twb = twa.detach().clone().requires_grad_(True)

    out_base = baseline_fn(xa, wa, ba, twa)
    out_kern = kernel_fn(xb, wb, bb, twb)

    grad = torch.randn_like(out_base)
    out_base.backward(grad)
    out_kern.backward(grad)

    torch.testing.assert_close(xb.grad, xa.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(wb.grad, wa.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(bb.grad, ba.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(twb.grad, twa.grad, atol=1e-3, rtol=1e-3)
