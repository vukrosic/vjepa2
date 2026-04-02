"""Parity test for fused_mish_gate kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_mish_gate import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    gate = torch.randn(shape["gate"], dtype=dtype, device="cuda")
    expected = baseline_fn(x, gate)
    actual = kernel_fn(x, gate)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_backward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda", requires_grad=True)
    gate = torch.randn(shape["gate"], dtype=dtype, device="cuda", requires_grad=True)
    x_ker = x.detach().clone().requires_grad_(True)
    gate_ker = gate.detach().clone().requires_grad_(True)
    y_ker = kernel_fn(x_ker, gate_ker)
    y_bas = baseline_fn(x, gate)
    y_ker.sum().backward()
    y_bas.sum().backward()
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(x_ker.grad, x.grad, atol=atol, rtol=0)
    torch.testing.assert_close(gate_ker.grad, gate.grad, atol=atol, rtol=0)
