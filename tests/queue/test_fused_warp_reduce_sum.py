"""Parity test for fused_warp_reduce_sum kernel."""
import torch, pytest
from src.models.utils.kernels.fused_warp_reduce_sum import kernel_fn, baseline_fn, SHAPES

ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=torch.float32, device="cuda")
    expected = baseline_fn(x)
    actual = kernel_fn(x)
    torch.testing.assert_close(actual, expected, atol=ATOL_FP32, rtol=ATOL_FP32)
