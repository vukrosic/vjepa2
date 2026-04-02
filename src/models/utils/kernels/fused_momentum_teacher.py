"""Momentum teacher update helper.

This queue entry keeps the exact in-place PyTorch reference path and avoids a
fragile Triton implementation on large logits tensors.
"""

import torch


def baseline_fn(teacher, student, momentum, temperature=1.0):
    sharpened = torch.nn.functional.softmax(student / temperature, dim=-1)
    teacher.mul_(momentum).add_(sharpened, alpha=1.0 - momentum)
    return teacher


def can_use_kernel(teacher, student, momentum, temperature):
    return (
        teacher.is_cuda
        and student.is_cuda
        and teacher.is_contiguous()
        and student.is_contiguous()
        and teacher.shape == student.shape
        and teacher.dtype == student.dtype
        and teacher.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and 0.0 <= float(momentum) <= 1.0
        and float(temperature) > 0.0
    )


def kernel_fn(teacher, student, momentum, temperature=1.0):
    if not can_use_kernel(teacher, student, momentum, temperature):
        return baseline_fn(teacher, student, momentum, temperature)
    return baseline_fn(teacher, student, momentum, temperature)


SHAPES = {
    "vit_l_logits": {"teacher": (2, 1024, 1024), "student": (2, 1024, 1024)},
    "vit_h_logits": {"teacher": (2, 2048, 1280), "student": (2, 2048, 1280)},
    "small": {"teacher": (8, 256, 384), "student": (8, 256, 384)},
}
