import torch
import pytest
from src.models.utils.kernels.fused_index_select_mean import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    indices = torch.randint(0, shape["x"][1], shape["indices_shape"], device="cuda")
    if not x.is_cuda:
        pytest.skip("CUDA not available")
    y_ker = kernel_fn(x, indices)
    y_bas = baseline_fn(x, indices)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(y_ker, y_bas, atol=atol, rtol=1e-2)
