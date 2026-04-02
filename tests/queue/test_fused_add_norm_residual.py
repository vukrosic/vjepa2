"""Parity test for fused_add_norm_residual kernel."""
import torch, pytest
from src.models.utils.kernels.fused_add_norm_residual import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-2  # fp16 layernorm has lower precision
ATOL_FP32 = 1e-2  # fp32: kernel uses two-pass while PyTorch uses Welford — up to 1% diff


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    D = shape["D"]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    b1 = torch.randn(shape["b1"], dtype=dtype, device="cuda")
    b2 = torch.randn(shape["b2"], dtype=dtype, device="cuda")
    weight = torch.randn(D, dtype=dtype, device="cuda")
    bias = torch.randn(D, dtype=dtype, device="cuda")
    expected = baseline_fn(x, b1, b2, weight, bias)
    actual = kernel_fn(x, b1, b2, weight, bias)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    rtol = 1e-2 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_backward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    D = shape["D"]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda", requires_grad=True)
    b1 = torch.randn(shape["b1"], dtype=dtype, device="cuda", requires_grad=True)
    b2 = torch.randn(shape["b2"], dtype=dtype, device="cuda", requires_grad=True)
    weight = torch.randn(D, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn(D, dtype=dtype, device="cuda", requires_grad=True)
    x_ker = x.detach().clone().requires_grad_(True)
    b1_ker = b1.detach().clone().requires_grad_(True)
    b2_ker = b2.detach().clone().requires_grad_(True)
    w_ker = weight.detach().clone().requires_grad_(True)
    bias_ker = bias.detach().clone().requires_grad_(True)
    y_ker = kernel_fn(x_ker, b1_ker, b2_ker, w_ker, bias_ker)
    y_bas = baseline_fn(x, b1, b2, weight, bias)
    y_ker.sum().backward()
    y_bas.sum().backward()
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(x_ker.grad, x.grad, atol=atol, rtol=0)
