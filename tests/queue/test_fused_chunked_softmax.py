"""Parity test for fused Chunked Softmax kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_chunked_softmax import can_use_kernel, kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 1e-2
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    chunk_size = shape["chunk_size"]
    expected = baseline_fn(x, chunk_size)
    actual = kernel_fn(x, chunk_size)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


def test_strict_fallback_for_noncontiguous_input():
    shape = SHAPES["small"]
    x = torch.randn(*shape["x"], dtype=torch.float32, device="cuda")
    x_nc = x.transpose(1, 2).contiguous().transpose(1, 2)
    chunk_size = shape["chunk_size"]
    assert not can_use_kernel(x_nc, chunk_size)
    torch.testing.assert_close(
        kernel_fn(x_nc, chunk_size),
        baseline_fn(x_nc, chunk_size),
        atol=ATOL_FP32,
        rtol=0,
    )
