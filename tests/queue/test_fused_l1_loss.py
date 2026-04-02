"""Parity test for fused_l1_loss kernel."""
import torch, pytest
from src.models.utils.kernels.fused_l1_loss import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5

@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_parity(shape_name, dtype):
    if not torch.cuda.is_available(): pytest.skip("CUDA not available")
    s = SHAPES[shape_name]
    pred = torch.randn(s["pred"], dtype=dtype, device="cuda")
    target = torch.randn(s["target"], dtype=dtype, device="cuda")
    expected = baseline_fn(pred, target)
    actual = kernel_fn(pred, target)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
