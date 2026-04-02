"""Parity test for fused_gating kernel."""
import torch, pytest
from src.models.utils.kernels.fused_gating import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    gate = torch.randn(*shape["gate"], dtype=dtype, device="cuda")
    input = torch.randn(*shape["input"], dtype=dtype, device="cuda")
    expected = baseline_fn(gate, input)
    actual = kernel_fn(gate, input)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    g1 = torch.randn(*shape["gate"], dtype=torch.float32, device="cuda", requires_grad=True)
    g2 = g1.detach().clone().requires_grad_(True)
    i1 = torch.randn(*shape["input"], dtype=torch.float32, device="cuda", requires_grad=True)
    i2 = i1.detach().clone().requires_grad_(True)
    out1 = baseline_fn(g1, i1)
    out2 = kernel_fn(g2, i2)
    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    torch.testing.assert_close(g2.grad, g1.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(i2.grad, i1.grad, atol=1e-3, rtol=1e-3)
