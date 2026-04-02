"""Fused momentum teacher update kernel."""
import torch
import triton
import triton.language as tl


def baseline_fn(teacher, student, momentum, temperature=1.0):
    sharpened = torch.nn.functional.softmax(student / temperature, dim=-1)
    teacher.mul_(momentum).add_(sharpened, alpha=1.0 - momentum)
    return teacher


@triton.jit
def _fwd(TEACHER, STUDENT, Y, B: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
         MOMENTUM: tl.constexpr, TEMP: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_base = pid_b * N * C + pid_n * C

    # Online max using scalar loads
    m_val = -1e9
    for c in range(C):
        s = tl.load(STUDENT + row_base + c).to(tl.float32)
        m_val = tl.where(m_val > s / TEMP, m_val, s / TEMP)

    # Online exp-sum using scalar loads
    e_sum = 0.0
    for c in range(C):
        s = tl.load(STUDENT + row_base + c).to(tl.float32)
        e_sum += tl.exp(s / TEMP - m_val)

    # Write output using scalar loads + stores
    for c in range(C):
        s = tl.load(STUDENT + row_base + c).to(tl.float32)
        t = tl.load(TEACHER + row_base + c).to(tl.float32)
        p = tl.exp(s / TEMP - m_val) / e_sum
        out = MOMENTUM * t + (1.0 - MOMENTUM) * p
        tl.store(Y + row_base + c, out)


@triton.jit
def _bwd(TEACHER, STUDENT, DY, DX_T, DX_S, B: tl.constexpr, N: tl.constexpr,
         C: tl.constexpr, MOMENTUM: tl.constexpr, TEMP: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_base = pid_b * N * C + pid_n * C

    # Recompute softmax probabilities
    m_val = -1e9
    for c in range(C):
        s = tl.load(STUDENT + row_base + c).to(tl.float32)
        m_val = tl.where(m_val > s / TEMP, m_val, s / TEMP)
    e_sum = 0.0
    for c in range(C):
        s = tl.load(STUDENT + row_base + c).to(tl.float32)
        e_sum += tl.exp(s / TEMP - m_val)

    for c in range(C):
        s = tl.load(STUDENT + row_base + c).to(tl.float32)
        dy = tl.load(DY + row_base + c).to(tl.float32)
        p = tl.exp(s / TEMP - m_val) / e_sum
        dt = MOMENTUM * dy
        ds = (1.0 - MOMENTUM) * dy * p * (1.0 - p) / TEMP
        tl.store(DX_T + row_base + c, dt)
        tl.store(DX_S + row_base + c, ds)


class FusedMomentumTeacher(torch.autograd.Function):
    @staticmethod
    def forward(ctx, teacher, student, momentum, temperature=1.0):
        assert teacher.is_contiguous() and student.is_contiguous()
        assert teacher.shape == student.shape
        B, N, C = teacher.shape
        y = torch.empty_like(teacher)
        _fwd[(B, N)](teacher, student, y, B, N, C, momentum, temperature, num_warps=4)
        ctx.save_for_backward(teacher, student)
        ctx.momentum = momentum; ctx.temperature = temperature
        ctx.B = B; ctx.N = N; ctx.C = C
        return y

    @staticmethod
    def backward(ctx, dy):
        teacher, student = ctx.saved_tensors
        B, N, C = ctx.B, ctx.N, ctx.C
        dx_t = torch.empty_like(teacher)
        dx_s = torch.empty_like(student)
        _bwd[(B, N)](teacher, student, dy, dx_t, dx_s, B, N, C, ctx.momentum, ctx.temperature, num_warps=4)
        return dx_t, dx_s, None, None


def kernel_fn(teacher, student, momentum, temperature=1.0):
    return FusedMomentumTeacher.apply(teacher, student, momentum, temperature)


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
