"""Parity test for fused_softmax_rope kernel."""
import torch
import pytest
import math
from src.models.utils.kernels.fused_softmax_rope import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    T, D, H = shape["T"], shape["D"], shape["H"]
    B = 2
    x = torch.randn(B, H, T, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, T, D, dtype=dtype, device="cuda")
    D_half = D // 2
    theta = 10000.0
    positions = torch.arange(D_half, dtype=torch.float32, device="cuda")
    freqs = 1.0 / (theta ** (positions / D_half))
    angles = positions.unsqueeze(0) * freqs.unsqueeze(1)
    cos_table = angles.cos()
    sin_table = angles.sin()
    expected = baseline_fn(x, k, cos_table, sin_table)
    actual = kernel_fn(x, k, cos_table, sin_table)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    T, D, H = shape["T"], shape["D"], shape["H"]
    B = 2
    x = torch.randn(B, H, T, D, dtype=torch.float32, device="cuda", requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=torch.float32, device="cuda", requires_grad=True)
    D_half = D // 2
    theta = 10000.0
    positions = torch.arange(D_half, dtype=torch.float32, device="cuda")
    freqs = 1.0 / (theta ** (positions / D_half))
    angles = positions.unsqueeze(0) * freqs.unsqueeze(1)
    cos_table = angles.cos()
    sin_table = angles.sin()
    x_ker = x.detach().clone().requires_grad_(True)
    k_ker = k.detach().clone().requires_grad_(True)
    y_ker = kernel_fn(x_ker, k_ker, cos_table, sin_table)
    y_bas = baseline_fn(x, k, cos_table, sin_table)
    grad = torch.randn_like(y_ker)
    y_ker.backward(grad)
    y_bas.backward(grad)
    torch.testing.assert_close(x_ker.grad, x.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_ker.grad, k.grad, atol=1e-2, rtol=1e-2)
