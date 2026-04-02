"""Parity test for fused_index_select_mean kernel."""
import torch, pytest
from src.models.utils.kernels.fused_index_select_mean import can_use_kernel, kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x = torch.randn(shape["x"], dtype=dtype, device="cuda")
    indices = torch.randint(0, shape["x"][1], shape["indices_shape"], device="cuda")
    expected = baseline_fn(x, indices)
    actual = kernel_fn(x, indices)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def test_strict_fallback_on_non_contiguous_indices():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES["small"]
    x = torch.randn(*shape["x"], dtype=torch.float32, device="cuda")
    indices = torch.randint(0, shape["x"][1], shape["indices_shape"], device="cuda")
    indices_nc = indices.t().contiguous().t()
    assert not can_use_kernel(x, indices_nc)
    torch.testing.assert_close(
        kernel_fn(x, indices_nc),
        baseline_fn(x, indices_nc),
        atol=0.0,
        rtol=0,
    )
