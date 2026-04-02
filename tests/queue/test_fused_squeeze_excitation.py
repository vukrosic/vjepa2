import torch
import pytest
from src.models.utils.kernels.fused_squeeze_excitation import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    se = torch.randn(*shape["se"], dtype=dtype, device="cuda")
    if not x.is_cuda:
        pytest.skip("CUDA not available")
    y_ker = kernel_fn(x, se)
    y_bas = baseline_fn(x, se)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(y_ker, y_bas, atol=atol, rtol=1e-2)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_backward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda", requires_grad=True)
    se = torch.randn(*shape["se"], dtype=dtype, device="cuda", requires_grad=True)
    x_ker = x.detach().clone().requires_grad_(True)
    se_ker = se.detach().clone().requires_grad_(True)
    if not x.is_cuda:
        pytest.skip("CUDA not available")
    y_ker = kernel_fn(x_ker, se_ker)
    y_bas = baseline_fn(x, se)
    y_ker.sum().backward()
    y_bas.sum().backward()
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(x_ker.grad, x.grad, atol=atol, rtol=1e-2)
