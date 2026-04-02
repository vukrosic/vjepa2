"""Parity test for fused_sincos_embed kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_sincos_embed import kernel_fn, baseline_fn, SHAPES

# fp16 sin/cos has lower precision; use generous atol
ATOL_FP16 = 5e-2
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    shape = SHAPES[shape_name]
    N = shape["positions"][0]
    embed_dim = shape["embed_dim"]
    positions = torch.arange(N, device="cuda")
    # kernel returns fp16; baseline returns fp32 — compare with relaxed atol
    expected = baseline_fn(positions, embed_dim)
    actual = kernel_fn(positions, embed_dim)
    torch.testing.assert_close(actual.float(), expected.float(), atol=ATOL_FP16, rtol=0)
