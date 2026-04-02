"""Parity test for fused_token_scatter kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_token_scatter import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    src = torch.randn(shape["src"][0], shape["src"][1], shape["src"][2], dtype=dtype, device="cuda")
    total_tokens = shape["total_tokens"]
    B, M = src.shape[0], src.shape[1]
    D = shape["src"][2]
    indices = torch.randint(0, total_tokens, (B, M), device="cuda")
    expected = baseline_fn(src, indices, total_tokens, D)
    actual = kernel_fn(src, indices, total_tokens, D)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    src1 = torch.randn(shape["src"][0], shape["src"][1], shape["src"][2], dtype=torch.float32, device="cuda", requires_grad=True)
    src2 = src1.detach().clone().requires_grad_(True)
    total_tokens = shape["total_tokens"]
    B, M = src1.shape[0], src1.shape[1]
    D = shape["src"][2]
    torch.manual_seed(42)
    indices = torch.randint(0, total_tokens, (B, M), device="cuda")
    out1 = baseline_fn(src1, indices, total_tokens, D)
    out2 = kernel_fn(src2, indices, total_tokens, D)
    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    torch.testing.assert_close(src2.grad, src1.grad, atol=1e-3, rtol=1e-3)
