"""Parity test for fused_masked_fill_softmax kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_masked_fill_softmax import kernel_fn, baseline_fn, SHAPES

ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    shape = SHAPES[shape_name]
    B, H, N, _ = shape["scores"]
    scores = torch.randn(*shape["scores"], dtype=torch.float32, device="cuda") * 0.5
    # ~80% tokens kept, 20% masked
    mask = (torch.rand(B, 1, N, N, device="cuda") > 0.2)
    scale = shape["scale"]
    expected = baseline_fn(scores, mask, scale)
    actual = kernel_fn(scores, mask, scale)
    torch.testing.assert_close(actual, expected, atol=ATOL_FP32, rtol=ATOL_FP32)
