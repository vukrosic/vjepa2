"""Parity test for fused_elu_v2 kernel."""
import torch, pytest
from src.models.utils.kernels.fused_elu_v2 import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    alpha = shape.get("alpha", 1.0)
    expected = baseline_fn(x, alpha)
    actual = kernel_fn(x, alpha)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-3)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_backward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda", requires_grad=True)
    x_ker = x.detach().clone().requires_grad_(True)
    alpha = shape.get("alpha", 1.0)
    y_ker = kernel_fn(x_ker, alpha)
    y_bas = baseline_fn(x, alpha)
    y_ker.sum().backward()
    y_bas.sum().backward()
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(x_ker.grad, x.grad, atol=atol, rtol=1e-2)
