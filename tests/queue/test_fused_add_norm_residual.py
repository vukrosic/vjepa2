import torch
import pytest
from src.models.utils.kernels.fused_add_norm_residual import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    D = shape["D"]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    b1 = torch.randn(shape["b1"], dtype=dtype, device="cuda")
    b2 = torch.randn(shape["b2"], dtype=dtype, device="cuda")
    weight = torch.randn(D, dtype=dtype, device="cuda")
    bias = torch.randn(D, dtype=dtype, device="cuda")
    if not x.is_cuda:
        pytest.skip("CUDA not available")
    y_ker = kernel_fn(x, b1, b2, weight, bias)
    y_bas = baseline_fn(x, b1, b2, weight, bias)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(y_ker, y_bas, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_backward_parity(shape_name, dtype):
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
    if not x.is_cuda:
        pytest.skip("CUDA not available")
    y_ker = kernel_fn(x_ker, b1_ker, b2_ker, w_ker, bias_ker)
    y_bas = baseline_fn(x, b1, b2, weight, bias)
    y_ker.sum().backward()
    y_bas.sum().backward()
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(x_ker.grad, x.grad, atol=atol, rtol=1e-2)
