"""Parity test for fused_gradient_clip kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_gradient_clip import kernel_fn, baseline_fn, SHAPES

ATOL_FP32 = 1e-4


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    shape = SHAPES[shape_name]
    grad = torch.randn(*shape["grad"], dtype=torch.float32, device="cuda") * 0.01
    grads_baseline = [g.clone() for g in [grad]]
    grads_kernel = [g.clone() for g in [grad]]
    max_norm = 1.0
    _, norm_b = baseline_fn(grads_baseline, max_norm)
    _, norm_k = kernel_fn(grads_kernel, max_norm)
    torch.testing.assert_close(norm_k, norm_b, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(grads_kernel[0], grads_baseline[0], atol=1e-4, rtol=1e-4)
