"""Parity test for fused_l2_distance kernel."""
import torch, pytest
from src.models.utils.kernels.fused_l2_distance import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    a = torch.randn(shape["a"], dtype=dtype, device="cuda")
    b = torch.randn(shape["b"], dtype=dtype, device="cuda")
    expected = baseline_fn(a, b)
    actual = kernel_fn(a, b)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
