"""Parity test for fused_sub_scale kernel."""
import torch, pytest
from src.models.utils.kernels.fused_sub_scale import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    y = torch.randn(shape["x"], dtype=dtype, device="cuda")
    expected = baseline_fn(x, y, 2.0)
    actual = kernel_fn(x, y, 2.0)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x1 = torch.randn(shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1 = torch.randn(shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    y2 = y1.detach().clone().requires_grad_(True)
    out1 = baseline_fn(x1, y1, 2.0)
    out2 = kernel_fn(x2, y2, 2.0)
    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    torch.testing.assert_close(x2.grad, x1.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(y2.grad, y1.grad, atol=1e-3, rtol=1e-3)
