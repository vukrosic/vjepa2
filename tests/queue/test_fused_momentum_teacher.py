"""Parity test for fused_momentum_teacher kernel."""
import torch, pytest
from src.models.utils.kernels.fused_momentum_teacher import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES[shape_name]
    teacher = torch.randn(shape["teacher"], dtype=dtype, device="cuda")
    student = torch.randn(shape["student"], dtype=dtype, device="cuda")
    momentum = 0.996
    temperature = 0.07
    expected = baseline_fn(teacher.clone(), student, momentum, temperature)
    actual = kernel_fn(teacher, student, momentum, temperature)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
