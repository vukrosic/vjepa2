"""Parity test for fused_attn_transpose kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_attn_transpose import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    B, N, H, D = shape["B"], shape["N"], shape["H"], shape["D"]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    expected = baseline_fn(x, B, N, H, D)
    actual = kernel_fn(x, B, N, H, D)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    B, N, H, D = shape["B"], shape["N"], shape["H"], shape["D"]
    xa = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)

    out_base = baseline_fn(xa, B, N, H, D)
    out_kern = kernel_fn(xb, B, N, H, D)

    grad = torch.randn_like(out_base)
    out_base.backward(grad)
    out_kern.backward(grad)

    torch.testing.assert_close(xb.grad, xa.grad, atol=1e-4, rtol=1e-4)
