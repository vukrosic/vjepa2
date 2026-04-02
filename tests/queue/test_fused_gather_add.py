"""Parity test for fused_gather_add kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_gather_add import can_use_kernel, kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"][0], shape["x"][1], shape["x"][2], dtype=dtype, device="cuda")
    B, M = shape["x"][0], shape["x"][1]
    D = shape["x"][2]
    total_tokens = shape["x"][1]  # indices range over the full sequence length
    torch.manual_seed(42)
    indices = torch.randint(0, total_tokens, (B, M), device="cuda")
    accum = torch.randn(B, M, D, dtype=dtype, device="cuda")
    expected = baseline_fn(x, indices, accum)
    actual = kernel_fn(x, indices, accum)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    x1 = torch.randn(shape["x"][0], shape["x"][1], shape["x"][2], dtype=torch.float32, device="cuda", requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    B, M = shape["x"][0], shape["x"][1]
    D = shape["x"][2]
    total_tokens = shape["x"][1]  # indices range over the full sequence length
    torch.manual_seed(42)
    indices = torch.randint(0, total_tokens, (B, M), device="cuda")
    accum1 = torch.randn(B, M, D, dtype=torch.float32, device="cuda", requires_grad=True)
    accum2 = accum1.detach().clone().requires_grad_(True)
    out1 = baseline_fn(x1, indices, accum1)
    out2 = kernel_fn(x2, indices, accum2)
    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    torch.testing.assert_close(x2.grad, x1.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(accum2.grad, accum1.grad, atol=1e-3, rtol=1e-3)


def test_strict_fallback_on_non_contiguous_inputs():
    shape = SHAPES["small"]
    B, N, D = shape["x"]
    x = torch.randn(B, N, D, dtype=torch.float32, device="cuda")
    indices = torch.randint(0, N, (B, shape["indices_shape"][1]), device="cuda", dtype=torch.long)
    accum = torch.randn(B, shape["indices_shape"][1], D, dtype=torch.float32, device="cuda")

    x_nc = x.transpose(1, 2).contiguous().transpose(1, 2)
    assert not can_use_kernel(x_nc, indices, accum)
    torch.testing.assert_close(
        kernel_fn(x_nc, indices, accum),
        baseline_fn(x_nc, indices, accum),
        atol=0.0,
        rtol=0,
    )
