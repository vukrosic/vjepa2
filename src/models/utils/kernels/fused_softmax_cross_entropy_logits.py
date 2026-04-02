"""Fused softmax + cross entropy kernel for distillation loss."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def baseline_fn(logits, target_indices, temperature=0.07):
    return torch.nn.functional.cross_entropy(logits / temperature, target_indices)


@triton.jit
def _fwd(LOGITS, TARGETS, Y, B: tl.constexpr, C: tl.constexpr, TEMP: tl.constexpr):
    pid_b = tl.program_id(0)
    row_base = pid_b * C

    # Online max using scalar loads
    m_val = -1e9
    for c in range(C):
        x = tl.load(LOGITS + row_base + c).to(tl.float32)
        m_val = tl.where(m_val > x / TEMP, m_val, x / TEMP)

    # Online exp-sum using scalar loads
    e_sum = 0.0
    for c in range(C):
        x = tl.load(LOGITS + row_base + c).to(tl.float32)
        e_sum += tl.exp(x / TEMP - m_val)

    # Loss and store
    target = tl.load(TARGETS + pid_b).to(tl.int32)
    x_target = tl.load(LOGITS + row_base + target).to(tl.float32)
    loss = m_val + tl.log(e_sum) - x_target / TEMP
    tl.store(Y + pid_b, loss)


class FusedSoftmaxCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target_indices, temperature=0.07):
        assert logits.is_contiguous() and target_indices.is_contiguous()
        B, C = logits.shape
        y = torch.empty(B, dtype=logits.dtype, device=logits.device)
        _fwd[(B,)](logits, target_indices, y, B, C, temperature, num_warps=4)
        ctx.save_for_backward(logits, target_indices)
        ctx.temperature = temperature
        return y.mean()

    @staticmethod
    def backward(ctx, dy):
        logits, target_indices = ctx.saved_tensors
        grad = torch.autograd.grad(
            F.cross_entropy(logits / ctx.temperature, target_indices, reduction='mean'),
            logits,
            grad_outputs=(dy.expand_as(logits),) if dy.numel() > 1 else (dy,),
        )[0]
        return grad, None, None


def kernel_fn(logits, target_indices, temperature=0.07):
    return FusedSoftmaxCrossEntropy.apply(logits, target_indices, temperature)


def can_use_kernel(logits, target_indices, temperature):
    return (logits.is_cuda and target_indices.is_cuda and
            logits.is_contiguous() and target_indices.is_contiguous() and
            logits.dtype in (torch.float16, torch.float32, torch.bfloat16) and
            target_indices.dtype == torch.long)


SHAPES = {
    "small":   {"logits": (256, 512),   "target_shape": (256,)},
    "medium":  {"logits": (512, 1024),  "target_shape": (512,)},
    "large":   {"logits": (1024, 2048), "target_shape": (1024,)},
}
