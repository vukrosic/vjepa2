"""Parity test for fused_3d_sincos_embed kernel."""
import torch, pytest
from src.models.utils.kernels.fused_3d_sincos_embed import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-2  # fp16 sin/cos has limited precision


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    shape = SHAPES[shape_name]
    expected = baseline_fn(**shape)
    actual = kernel_fn(**shape)
    torch.testing.assert_close(actual.float(), expected.float(), atol=ATOL_FP16, rtol=1e-2)
