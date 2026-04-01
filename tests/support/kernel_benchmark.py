from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class AttentionCase:
    name: str
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    masked: bool = False
    causal: bool = False


@dataclass(frozen=True)
class KernelResult:
    case: str
    policy: str
    masked: bool
    dtype: str
    shape: tuple[int, int, int, int]
    backend_order: tuple[str, ...]
    median_ms: float
    mean_ms: float
    min_ms: float
    samples_ms: tuple[float, ...]


def make_attention_inputs(case: AttentionCase, device: str = "cuda"):
    batch, heads, tokens, head_dim = case.shape
    q = torch.randn(batch, heads, tokens, head_dim, device=device, dtype=case.dtype)
    k = torch.randn(batch, heads, tokens, head_dim, device=device, dtype=case.dtype)
    v = torch.randn(batch, heads, tokens, head_dim, device=device, dtype=case.dtype)
    attn_mask = None
    if case.masked:
        attn_mask = torch.ones(tokens, tokens, device=device, dtype=torch.bool).tril()
    return q, k, v, attn_mask


def _backend_order(policy: str, q: torch.Tensor, attn_mask: torch.Tensor | None):
    if q.device.type != "cuda":
        return (SDPBackend.MATH,)

    if policy == "default_auto":
        return ()

    if policy == "math_only":
        return (SDPBackend.MATH,)

    if policy == "optimized":
        if attn_mask is not None:
            return (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH)
        if q.dtype in (torch.float16, torch.bfloat16) and torch.cuda.get_device_capability(q.device) >= (8, 0):
            return (
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.MATH,
            )
        return (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH)

    raise ValueError(f"Unknown policy: {policy}")


def run_attention_policy(
    policy: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
):
    backend_order = _backend_order(policy, q, attn_mask)

    if policy == "default_auto":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

    if policy == "math_only":
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

    if policy == "optimized":
        with sdpa_kernel(list(backend_order), set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

    raise ValueError(f"Unknown policy: {policy}")


def benchmark_policy(
    case: AttentionCase,
    policy: str,
    warmup_iters: int,
    timed_iters: int,
    repeats: int,
    device: str = "cuda",
    check_parity: bool = False,
):
    q, k, v, attn_mask = make_attention_inputs(case, device=device)
    backend_order = _backend_order(policy, q, attn_mask)

    if check_parity:
        baseline = run_attention_policy("default_auto", q, k, v, attn_mask=attn_mask, is_causal=case.causal)
        candidate = run_attention_policy(policy, q, k, v, attn_mask=attn_mask, is_causal=case.causal)
        tol = 2e-3 if case.dtype in (torch.float16, torch.bfloat16) else 1e-5
        torch.testing.assert_close(candidate, baseline, atol=tol, rtol=tol)

    for _ in range(warmup_iters):
        run_attention_policy(policy, q, k, v, attn_mask=attn_mask, is_causal=case.causal)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(timed_iters):
            run_attention_policy(policy, q, k, v, attn_mask=attn_mask, is_causal=case.causal)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / timed_iters)

    return KernelResult(
        case=case.name,
        policy=policy,
        masked=case.masked,
        dtype=str(case.dtype).replace("torch.", ""),
        shape=case.shape,
        backend_order=tuple(backend.name for backend in backend_order) if backend_order else ("DEFAULT_AUTO",),
        median_ms=median(samples),
        mean_ms=mean(samples),
        min_ms=min(samples),
        samples_ms=tuple(samples),
    )


def summarize_speedups(results: list[KernelResult], baseline_policy: str = "default_auto"):
    rows = []
    grouped: dict[str, dict[str, KernelResult]] = {}
    for result in results:
        grouped.setdefault(result.case, {})[result.policy] = result

    for case_name, per_policy in grouped.items():
        baseline = per_policy[baseline_policy]
        for policy_name, result in per_policy.items():
            speedup = baseline.median_ms / result.median_ms
            delta_pct = 100.0 * (baseline.median_ms - result.median_ms) / baseline.median_ms
            rows.append(
                {
                    "case": case_name,
                    "policy": policy_name,
                    "masked": result.masked,
                    "dtype": result.dtype,
                    "shape": "x".join(str(v) for v in result.shape),
                    "median_ms": result.median_ms,
                    "mean_ms": result.mean_ms,
                    "min_ms": result.min_ms,
                    "baseline_ms": baseline.median_ms,
                    "speedup_x": speedup,
                    "delta_pct": delta_pct,
                    "backend_order": " > ".join(result.backend_order),
                }
            )
    rows.sort(key=lambda row: (row["case"], row["policy"]))
    return rows


def format_speedup_table(rows: list[dict[str, object]], baseline_policy: str = "default_auto"):
    header = [
        "case",
        "policy",
        "median_ms",
        "baseline_ms",
        "speedup_x",
        "delta_pct",
        "backend_order",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["policy"]),
                    f'{row["median_ms"]:.4f}',
                    f'{row["baseline_ms"]:.4f}',
                    f'{row["speedup_x"]:.3f}x',
                    f'{row["delta_pct"]:+.2f}%',
                    str(row["backend_order"]),
                ]
            )
            + " |"
        )
    lines.append(f"\nBaseline policy: `{baseline_policy}`")
    return "\n".join(lines)


def default_attention_cases(batch: int, heads: int, tokens: int, head_dim: int, dtype: str):
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    shape = (batch, heads, tokens, head_dim)
    return [
        AttentionCase(name="unmasked", shape=shape, dtype=torch_dtype, masked=False),
        AttentionCase(name="masked", shape=shape, dtype=torch_dtype, masked=True),
    ]


def default_policies():
    return ["optimized", "default_auto", "math_only"]
