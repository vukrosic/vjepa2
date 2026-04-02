"""Parity test for fused_pow_scalar kernel."""
import torch, pytest
from src.models.utils.kernels.fused_pow_scalar import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda").abs() + 0.1
    exp = shape.get("exponent", 2.0)
    expected = baseline_fn(x, exp)
    actual = kernel_fn(x, exp)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-3)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_backward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda", requires_grad=True).abs() + 0.1
    x_ker = x.detach().clone().requires_grad_(True)
    exp = shape.get("exponent", 2.0)
    y_ker = kernel_fn(x_ker, exp)
    y_bas = baseline_fn(x, exp)
    y_ker.sum().backward()
    y_bas.sum().backward()
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(x_ker.grad, x.grad, atol=atol, rtol=1e-2)
