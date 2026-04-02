"""Parity test for fused_rope_apply kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_rope_apply import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-2
ATOL_FP32 = 1e-3  # RoPE has minor precision differences due to fp16 sin/cos


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=torch.float32, device="cuda") * 0.5
    pos = torch.arange(shape["pos"][0], dtype=torch.float32, device="cuda")
    head_dim = shape["head_dim"]
    expected = baseline_fn(x, pos, head_dim)
    actual = kernel_fn(x, pos, head_dim)
    torch.testing.assert_close(actual, expected, atol=ATOL_FP32, rtol=ATOL_FP32)
