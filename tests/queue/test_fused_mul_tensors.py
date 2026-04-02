"""Parity test for fused_mul_tensors kernel."""
import torch, pytest
from src.models.utils.kernels.fused_mul_tensors import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
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
    x1b = x1a.detach().clone().requires_grad_(True)
    x2a = torch.randn(*shape["x2"], dtype=torch.float32, device="cuda", requires_grad=True)
    x2b = x2a.detach().clone().requires_grad_(True)
    out1 = baseline_fn(x1a, x2a)
    out2 = kernel_fn(x1b, x2b)
    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    torch.testing.assert_close(x1b.grad, x1a.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(x2b.grad, x2a.grad, atol=1e-3, rtol=1e-3)
