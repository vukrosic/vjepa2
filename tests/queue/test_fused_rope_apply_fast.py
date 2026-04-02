"""Parity test for fused_rope_apply_fast kernel."""
import torch, pytest
from src.models.utils.kernels.fused_rope_apply_fast import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    pos = torch.arange(shape["pos"][0], dtype=dtype, device="cuda")
    head_dim = shape["head_dim"]
    expected = baseline_fn(x, pos, head_dim)
    actual = kernel_fn(x, pos, head_dim)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
