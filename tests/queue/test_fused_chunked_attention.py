"""Parity test for fused_chunked_attention kernel."""
import torch, pytest
from src.models.utils.kernels.fused_chunked_attention import kernel_fn, baseline_fn, SHAPES

ATOL_FP32 = 1e-3


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    shape = SHAPES[shape_name]
    q = torch.randn(*shape["q"], dtype=torch.float32, device="cuda")
    k = torch.randn(*shape["k"], dtype=torch.float32, device="cuda")
    v = torch.randn(*shape["v"], dtype=torch.float32, device="cuda")
    scale = shape["scale"]
    expected = baseline_fn(q, k, v, scale)
    actual = kernel_fn(q, k, v, scale)
    torch.testing.assert_close(actual, expected, atol=ATOL_FP32, rtol=ATOL_FP32)
