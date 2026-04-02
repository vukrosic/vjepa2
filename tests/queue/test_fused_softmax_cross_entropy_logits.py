"""Parity test for fused_softmax_cross_entropy_logits kernel."""
import torch, pytest
from src.models.utils.kernels.fused_softmax_cross_entropy_logits import kernel_fn, baseline_fn, SHAPES

ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    logits = torch.randn(shape["logits"], dtype=torch.float32, device="cuda")
    targets = torch.randint(0, shape["logits"][1], shape["target_shape"], device="cuda")
    temperature = 0.07
    expected = baseline_fn(logits, targets, temperature)
    actual = kernel_fn(logits, targets, temperature)
    torch.testing.assert_close(actual, expected, atol=ATOL_FP32, rtol=0)
