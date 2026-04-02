"""Parity test for fused SiLU * multiply kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_silu_mul import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2  # SiLU has fp16 precision quirks
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x1 = torch.randn(*shape["x1"], dtype=dtype, device="cuda")
    x2 = torch.randn(*shape["x2"], dtype=dtype, device="cuda")
    expected = baseline_fn(x1, x2)
    actual = kernel_fn(x1, x2)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    x1a = torch.randn(*shape["x1"], dtype=torch.float32, device="cuda", requires_grad=True)
    x2a = torch.randn(*shape["x2"], dtype=torch.float32, device="cuda", requires_grad=True)
    x1b = x1a.detach().clone().requires_grad_(True)
    x2b = x2a.detach().clone().requires_grad_(True)

    out_base = baseline_fn(x1a, x2a)
    out_kern = kernel_fn(x1b, x2b)

    grad = torch.randn_like(out_base)
    out_base.backward(grad)
    out_kern.backward(grad)

    torch.testing.assert_close(x1b.grad, x1a.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(x2b.grad, x2a.grad, atol=1e-4, rtol=1e-4)
