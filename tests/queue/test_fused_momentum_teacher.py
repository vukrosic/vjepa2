"""Parity test for fused_momentum_teacher kernel."""
import torch, pytest
from src.models.utils.kernels.fused_momentum_teacher import can_use_kernel, kernel_fn, baseline_fn, SHAPES

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


def test_strict_fallback_on_non_contiguous():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shape = SHAPES["small"]
    teacher = torch.randn(shape["teacher"], dtype=torch.float32, device="cuda")
    student = torch.randn(shape["student"], dtype=torch.float32, device="cuda")
    teacher_nc = teacher.transpose(1, 2).contiguous().transpose(1, 2)
    assert not can_use_kernel(teacher_nc, student, 0.996, 0.07)
    expected = baseline_fn(teacher_nc.clone(), student, 0.996, 0.07)
    actual = kernel_fn(teacher_nc, student, 0.996, 0.07)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0)
