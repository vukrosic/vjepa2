"""Parity test for fused_mlp_block kernel."""
import torch, pytest
from src.models.utils.kernels.fused_mlp_block import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    residual = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    D = shape["D"]
    ln_weight = torch.ones(D, dtype=dtype, device="cuda")
    ln_bias = torch.zeros(D, dtype=dtype, device="cuda")
    proj1_weight = torch.randn(D * 4, D, dtype=dtype, device="cuda")
    proj1_bias = torch.zeros(D * 4, dtype=dtype, device="cuda")
    expected = baseline_fn(x, residual, ln_weight, ln_bias, proj1_weight, proj1_bias)
    actual = kernel_fn(x, residual, ln_weight, ln_bias, proj1_weight, proj1_bias)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
