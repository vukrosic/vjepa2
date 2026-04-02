"""Parity test for fused_sum_square kernel."""
import torch, pytest
from src.models.utils.kernels.fused_sum_square import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    expected_sum, expected_sq = baseline_fn(x)
    actual_sum, actual_sq = kernel_fn(x)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual_sum.float(), expected_sum.float(), atol=atol, rtol=1e-3)
    torch.testing.assert_close(actual_sq.float(), expected_sq.float(), atol=atol, rtol=1e-3)
