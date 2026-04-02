"""Parity test for fused_online_softmax kernel."""
import torch, pytest
from src.models.utils.kernels.fused_online_softmax import kernel_fn, baseline_fn, SHAPES

ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    shape = SHAPES[shape_name]
    scores = torch.randn(*shape["scores"], dtype=torch.float32, device="cuda") * 0.5
    expected = baseline_fn(scores)
    actual = kernel_fn(scores)
    torch.testing.assert_close(actual, expected, atol=ATOL_FP32, rtol=ATOL_FP32)
