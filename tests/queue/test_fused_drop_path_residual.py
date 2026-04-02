"""Parity test for fused_drop_path_residual kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_drop_path_residual import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    # Use training=False, drop_prob=0.0 for deterministic comparison (output = x + y)
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    y = torch.randn(*shape["y"], dtype=dtype, device="cuda")
    expected = baseline_fn(x, y, drop_prob=0.0, training=False)
    actual = kernel_fn(x, y, drop_prob=0.0, training=False)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    xa = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    ya = torch.randn(*shape["y"], dtype=torch.float32, device="cuda", requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    yb = ya.detach().clone().requires_grad_(True)

    out_base = baseline_fn(xa, ya, drop_prob=0.0, training=False)
    out_kern = kernel_fn(xb, yb, drop_prob=0.0, training=False)

    grad = torch.randn_like(out_base)
    out_base.backward(grad)
    out_kern.backward(grad)

    torch.testing.assert_close(xb.grad, xa.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(yb.grad, ya.grad, atol=1e-4, rtol=1e-4)
