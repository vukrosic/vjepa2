"""Parity test for fused_cross_entropy kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_cross_entropy import kernel_fn, baseline_fn, SHAPES

ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    """Cross-entropy is numerically sensitive — fp32 only."""
    shape = SHAPES[shape_name]
    logits = torch.randn(*shape["logits"], dtype=torch.float32, device="cuda")
    labels = torch.randint(0, shape["logits"][1], shape["labels_shape"], device="cuda").long()
    temperature = shape["temperature"]
    expected = baseline_fn(logits, labels, temperature)
    actual = kernel_fn(logits, labels, temperature)
    torch.testing.assert_close(actual, expected, atol=ATOL_FP32, rtol=ATOL_FP32)
