"""Parity test for fused_adamw_step kernel."""
import torch
import pytest
from src.models.utils.kernels.fused_adamw_step import kernel_fn, baseline_fn, SHAPES

ATOL_FP32 = 1e-4  # optimizer steps accumulate floating point errors


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    """Test that fused AdamW matches PyTorch reference (fp32 only — optimizer state is fp32)."""
    shape = SHAPES[shape_name]
    param_shape = shape["param"]

    param1 = torch.randn(*param_shape, dtype=torch.float32, device="cuda")
    param2 = param1.clone()
    grad = torch.randn(*param_shape, dtype=torch.float32, device="cuda") * 0.01
    exp_avg1 = torch.randn(*param_shape, dtype=torch.float32, device="cuda") * 0.01
    exp_avg2 = exp_avg1.clone()
    exp_avg_sq1 = torch.rand(*param_shape, dtype=torch.float32, device="cuda") * 0.001
    exp_avg_sq2 = exp_avg_sq1.clone()

    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01
    step = 100

    baseline_fn(param1, grad.clone(), exp_avg1, exp_avg_sq1, lr, beta1, beta2, eps, weight_decay, step)
    kernel_fn(param2, grad.clone(), exp_avg2, exp_avg_sq2, lr, beta1, beta2, eps, weight_decay, step)

    torch.testing.assert_close(param2, param1, atol=ATOL_FP32, rtol=1e-4)
    torch.testing.assert_close(exp_avg2, exp_avg1, atol=ATOL_FP32, rtol=1e-4)
    torch.testing.assert_close(exp_avg_sq2, exp_avg_sq1, atol=ATOL_FP32, rtol=1e-4)
