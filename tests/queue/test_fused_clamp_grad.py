"""Parity test for fused_clamp_grad kernel."""
import torch, pytest
from src.models.utils.kernels.fused_clamp_grad import kernel_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=torch.float32, device="cuda")
    expected = x  # identity pass-through
    actual = kernel_fn(x, -1.0, 1.0)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    dy = torch.randn(*shape["x"], dtype=torch.float32, device="cuda")
    dx_ker = kernel_fn(x, dy, -1.0, 1.0)
    dx_ref = torch.clamp_backward(dy, x, -1.0, 1.0)
    torch.testing.assert_close(dx_ker, dx_ref, atol=1e-3, rtol=1e-3)
