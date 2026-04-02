"""Parity test for fused Lp loss kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_loss import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    s = SHAPES[shape_name]
    z = torch.randn(*s["z"], dtype=dtype, device="cuda")
    h = torch.randn(*s["h"], dtype=dtype, device="cuda")
    expected = baseline_fn(z, h, s["loss_exp"])
    actual = kernel_fn(z, h, s["loss_exp"])
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=1e-3)
