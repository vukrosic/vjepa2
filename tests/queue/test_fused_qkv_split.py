"""Parity test for fused_qkv_split kernel."""
import torch
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.models.utils.kernels.fused_qkv_split import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    B, N, H, D = shape["B"], shape["N"], shape["H"], shape["D"]
    qkv_linear = torch.randn(*shape["qkv_linear"], dtype=dtype, device="cuda")
    q_exp, k_exp, v_exp = baseline_fn(qkv_linear, B, N, H, D)
    q_act, k_act, v_act = kernel_fn(qkv_linear, B, N, H, D)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(q_act, q_exp, atol=atol, rtol=0)
    torch.testing.assert_close(k_act, k_exp, atol=atol, rtol=0)
    torch.testing.assert_close(v_act, v_exp, atol=atol, rtol=0)


@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    shape = SHAPES[shape_name]
    B, N, H, D = shape["B"], shape["N"], shape["H"], shape["D"]
    qkv_a = torch.randn(*shape["qkv_linear"], dtype=torch.float32, device="cuda", requires_grad=True)
    qkv_b = qkv_a.detach().clone().requires_grad_(True)

    q_exp, k_exp, v_exp = baseline_fn(qkv_a, B, N, H, D)
    q_act, k_act, v_act = kernel_fn(qkv_b, B, N, H, D)

    grad_q = torch.randn_like(q_exp)
    grad_k = torch.randn_like(k_exp)
    grad_v = torch.randn_like(v_exp)

    (q_exp * grad_q + k_exp * grad_k + v_exp * grad_v).sum().backward()
    (q_act * grad_q + k_act * grad_k + v_act * grad_v).sum().backward()

    torch.testing.assert_close(qkv_b.grad, qkv_a.grad, atol=1e-4, rtol=1e-4)
