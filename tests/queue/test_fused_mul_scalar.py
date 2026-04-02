"""Parity test for fused_mul_scalar kernel."""
import torch, pytest
from src.models.utils.kernels.fused_mul_scalar import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    scalar = shape.get("scalar", 0.1)
    expected = baseline_fn(x, scalar)
    actual = kernel_fn(x, scalar)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_backward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda", requires_grad=True)
    x_ker = x.detach().clone().requires_grad_(True)
    scalar = shape.get("scalar", 0.1)
    y_ker = kernel_fn(x_ker, scalar)
    y_bas = baseline_fn(x, scalar)
    y_ker.sum().backward()
    y_bas.sum().backward()
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(x_ker.grad, x.grad, atol=atol, rtol=0)
