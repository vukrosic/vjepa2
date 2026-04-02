"""Parity test for fused_layernorm_cat kernel."""
import torch, pytest
from src.models.utils.kernels.fused_layernorm_cat import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    num_splits = shape["num_splits"]
    D = shape["D"]
    weight = torch.ones(D, dtype=dtype, device="cuda")
    bias = torch.zeros(D, dtype=dtype, device="cuda")
    expected = baseline_fn(x, num_splits, weight, bias)
    actual = kernel_fn(x, weight, bias, num_splits)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
