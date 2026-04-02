"""Parity test for fused_softmax_temperature kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_softmax_temperature import kernel_fn, baseline_fn, SHAPES

ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    """Softmax is numerically sensitive — test fp32 only."""
    shape = SHAPES[shape_name]
    scores = torch.randn(*shape["scores"], dtype=torch.float32, device="cuda") * 0.5
    scale = shape["scale"]
    expected = baseline_fn(scores, scale)
    actual = kernel_fn(scores, scale)
    torch.testing.assert_close(actual, expected, atol=ATOL_FP32, rtol=ATOL_FP32)
