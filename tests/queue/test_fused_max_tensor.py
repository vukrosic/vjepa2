"""Parity test for fused_max_tensor kernel."""
import torch, pytest
from src.models.utils.kernels.fused_max_tensor import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5

@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available(): pytest.skip("CUDA not available")
    s = SHAPES[shape_name]
    a = torch.randn(s["a"], dtype=dtype, device="cuda")
    b = torch.randn(s["b"], dtype=dtype, device="cuda")
    expected = baseline_fn(a, b)
    actual = kernel_fn(a, b)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)

@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    if not torch.cuda.is_available(): pytest.skip("CUDA not available")
    s = SHAPES[shape_name]
    a1 = torch.randn(s["a"], dtype=torch.float32, device="cuda", requires_grad=True)
    a2 = a1.detach().clone().requires_grad_(True)
    b1 = torch.randn(s["b"], dtype=torch.float32, device="cuda", requires_grad=True)
    b2 = b1.detach().clone().requires_grad_(True)
    out1 = baseline_fn(a1, b1)
    out2 = kernel_fn(a2, b2)
    grad = torch.randn_like(out1)
    out1.backward(grad); out2.backward(grad)
    torch.testing.assert_close(a2.grad, a1.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(b2.grad, b1.grad, atol=1e-3, rtol=1e-3)
