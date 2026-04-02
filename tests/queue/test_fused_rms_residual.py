"""Parity test for fused RMSNorm + Residual Add kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_rms_residual import can_use_kernel, kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    residual = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    weight = torch.randn(shape["C"], dtype=dtype, device="cuda")
    expected = baseline_fn(x, residual, weight)
    actual = kernel_fn(x, residual, weight)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    xa = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    ra = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    wa = torch.randn(shape["C"], dtype=torch.float32, device="cuda", requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    rb = ra.detach().clone().requires_grad_(True)
    wb = wa.detach().clone().requires_grad_(True)

    out_base = baseline_fn(xa, ra, wa)
    out_kern = kernel_fn(xb, rb, wb)

    grad = torch.randn_like(out_base)
    out_base.backward(grad)
    out_kern.backward(grad)

    torch.testing.assert_close(xb.grad, xa.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(rb.grad, ra.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(wb.grad, wa.grad, atol=1e-3, rtol=1e-3)


def test_strict_fallback_on_non_contiguous():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES["small"]
    x = torch.randn(*shape["x"], device="cuda")
    residual = torch.randn(*shape["x"], device="cuda")
    weight = torch.randn(shape["C"], device="cuda")

    x_nc = x.transpose(1, 2).contiguous().transpose(1, 2)
    assert not can_use_kernel(x_nc, residual, weight)
    torch.testing.assert_close(
        kernel_fn(x_nc, residual, weight),
        baseline_fn(x_nc, residual, weight),
        atol=0.0,
        rtol=0,
    )
