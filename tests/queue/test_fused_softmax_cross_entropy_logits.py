import torch
import pytest
from src.models.utils.kernels.fused_softmax_cross_entropy_logits import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_forward_parity(shape_name):
    shape = SHAPES[shape_name]
    logits = torch.randn(*shape["logits"], dtype=torch.float32, device="cuda")
    targets = torch.randint(0, shape["logits"][1], shape["target_shape"], device="cuda")
    temperature = 0.07
    if not logits.is_cuda:
        pytest.skip("CUDA not available")
    y_ker = kernel_fn(logits, targets, temperature)
    y_bas = baseline_fn(logits, targets, temperature)
    torch.testing.assert_close(y_ker, y_bas, atol=ATOL_FP32, rtol=1e-2)
