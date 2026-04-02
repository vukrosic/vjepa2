"""Parity test for fused_exp kernel."""
import torch, pytest
from src.models.utils.kernels.fused_exp import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda") * 0.5  # keep values small
    expected = baseline_fn(x)
    actual = kernel_fn(x)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    x1 = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True) * 0.5
    x2 = x1.detach().clone().requires_grad_(True)
    out1 = baseline_fn(x1)
    out2 = kernel_fn(x2)
    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    torch.testing.assert_close(x2.grad, x1.grad, atol=1e-3, rtol=1e-3)
