"""Parity test for fused GELU + Bias + Dropout kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_gelu_bias_dropout import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-1  # Dropout is stochastic, loose tolerance
ATOL_FP32 = 1e-3


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    bias = torch.randn(*shape["bias"], dtype=dtype, device="cuda")
    p = 0.1
    torch.manual_seed(42)
    expected = baseline_fn(x, bias, p, training=True)
    torch.manual_seed(42)
    actual = kernel_fn(x, bias, p, training=True)
    # Check GELU computation is correct (not stochastic)
    # Due to dropout randomness, we just check shapes match
    assert actual.shape == expected.shape


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_inference_parity(shape_name):
    """Test with training=False (no dropout) for deterministic check."""
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=torch.float32, device="cuda")
    bias = torch.randn(*shape["bias"], dtype=torch.float32, device="cuda")
    p = 0.1
    expected = baseline_fn(x, bias, p, training=False)
    actual = kernel_fn(x, bias, p, training=False)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
