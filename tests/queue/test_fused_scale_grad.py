"""Parity test for fused_scale_grad kernel."""
import torch, pytest
from src.models.utils.kernels.fused_scale_grad import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    grad = torch.randn(*shape["grad"], dtype=dtype, device="cuda")
    scale = 0.1
    max_val = 1.0
    expected = baseline_fn(grad.clone(), scale, max_val)
    actual = kernel_fn(grad, scale, max_val)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
