"""Parity test for fused_layer_norm_1d kernel."""
import torch, pytest
from src.models.utils.kernels.fused_layer_norm_1d import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5

@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available(): pytest.skip("CUDA not available")
    s = SHAPES[shape_name]
    x = torch.randn(s["x"], dtype=dtype, device="cuda")
    w = torch.randn(s["x"][-1], dtype=dtype, device="cuda")
    b = torch.randn(s["x"][-1], dtype=dtype, device="cuda")
    expected = baseline_fn(x, w, b)
    actual = kernel_fn(x, w, b)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
