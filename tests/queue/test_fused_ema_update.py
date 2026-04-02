"""Parity test for fused_ema_update kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_ema_update import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5

MOMENTA = [0.996, 0.999]


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("momentum", MOMENTA)
def test_forward_parity(shape_name, dtype, momentum):
    shape = SHAPES[shape_name]
    student = torch.randn(*shape["student"], dtype=dtype, device="cuda")
    target_base = torch.randn(*shape["target"], dtype=dtype, device="cuda")
    # Clone so each call gets an independent copy before in-place update
    target_kern = target_base.clone()

    baseline_fn(target_base, student, momentum)
    kernel_fn(target_kern, student, momentum)

    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(target_kern, target_base, atol=atol, rtol=0)
