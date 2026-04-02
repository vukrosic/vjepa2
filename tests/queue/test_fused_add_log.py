"""Parity test for fused_add_log kernel."""
import torch, pytest
from src.models.utils.kernels.fused_add_log import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    other = torch.randn(*shape["other"], dtype=dtype, device="cuda").abs_() + 0.1
    expected = baseline_fn(x, other)
    actual = kernel_fn(x, other)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x1 = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    other1 = (torch.randn(*shape["other"], dtype=torch.float32, device="cuda").abs_() + 0.1).requires_grad_(True)
    other2 = other1.detach().clone().requires_grad_(True)
    out1 = baseline_fn(x1, other1)
    out2 = kernel_fn(x2, other2)
    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    torch.testing.assert_close(x2.grad, x1.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(other2.grad, other1.grad, atol=1e-3, rtol=1e-3)
