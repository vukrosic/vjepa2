"""Parity test for fused_sigmoid_bw kernel."""
import torch, pytest
from src.models.utils.kernels.fused_sigmoid_bw import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_backward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    x1 = torch.randn(*shape["x"], dtype=dtype, device="cuda", requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    dy = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    out1 = baseline_fn(x1, dy)
    out2 = kernel_fn(x2, dy)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(out2, out1, atol=atol, rtol=0)
