"""Fused momentum teacher update kernel.

Pattern: teacher = momentum * teacher + (1-momentum) * softmax(student / T)
Fuses: softmax + EMA update in one pass, avoiding materialization of softmax output.
"""
import torch
import triton
import triton.language as tl


# --- BASELINE (exact copy) ---
def baseline_fn(teacher, student, momentum, temperature=1.0):
    sharpened = torch.nn.functional.softmax(student / temperature, dim=-1)
    teacher.mul_(momentum).add_(sharpened, alpha=1.0 - momentum)
    return teacher


# --- KERNEL ---
@triton.jit
def _fused_momentum_teacher_fwd(TEACHER, STUDENT, Y, B: tl.constexpr, N: tl.constexpr, C: tl.constexpr, MOMENTUM: tl.constexpr, TEMP: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Phase 1: find max of student row
    row_base = pid_b * N * C + pid_n * C
    max_val = -1e9
    for c in range(C):
        s = tl.load(STUDENT + row_base + c, mask=mask_c, other=0.0).to(tl.float32)
        max_val = tl.max(max_val, s / TEMP)
        mask_c = (offs_c + c + 1) < C
    mask_c = offs_c < C

    # Phase 2: compute exp and sum
    exp_sum = 0.0
    for c in range(C):
        s = tl.load(STUDENT + row_base + c, mask=mask_c, other=0.0).to(tl.float32)
        exp_val = tl.exp(s / TEMP - max_val)
        exp_sum += exp_val
        mask_c = (offs_c + c + 1) < C
    mask_c = offs_c < C

    # Phase 3: write output: teacher = momentum * teacher + (1-momentum) * softmax
    for c in range(C):
        s = tl.load(STUDENT + row_base + c, mask=mask_c, other=0.0).to(tl.float32)
        t = tl.load(TEACHER + row_base + c, mask=mask_c, other=0.0).to(tl.float32)
        softmax_val = tl.exp(s / TEMP - max_val) / exp_sum
        out = MOMENTUM * t + (1.0 - MOMENTUM) * softmax_val
        tl.store(Y + row_base + c, out, mask=mask_c)
        mask_c = (offs_c + c + 1) < C


def kernel_fn(teacher, student, momentum, temperature=1.0):
    assert teacher.is_contiguous() and student.is_contiguous()
    assert teacher.shape == student.shape
    B, N, C = teacher.shape
    y = torch.empty_like(teacher)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (B, N)
    _fused_momentum_teacher_fwd[grid](teacher, student, y, B, N, C, momentum, temperature, BLOCK_C=BLOCK_C, num_warps=4)
    return y


def can_use_kernel(teacher, student, momentum, temperature):
    return (teacher.is_cuda and student.is_cuda and
            teacher.is_contiguous() and student.is_contiguous() and
            teacher.shape == student.shape and
            teacher.dtype == student.dtype and
            teacher.dtype in (torch.float16, torch.float32, torch.bfloat16))


SHAPES = {
    "vit_l_logits":  {"teacher": (2, 1024, 1024), "student": (2, 1024, 1024)},
    "vit_h_logits":  {"teacher": (2, 2048, 1280), "student": (2, 2048, 1280)},
    "small":         {"teacher": (8, 256, 384),   "student": (8, 256, 384)},
}
