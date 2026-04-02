"""Benchmark for fused_rotate_query_key_pair."""

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.utils.kernels.fused_rotate_query_key_pair import SHAPES, baseline_fn, kernel_fn


def bench_cuda(fn, warmup=20, iters=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


results = {}
for name, shape in SHAPES.items():
    q = torch.randn(*shape["q"], dtype=torch.float16, device="cuda")
    k = torch.randn(*shape["k"], dtype=torch.float16, device="cuda")
    pos = torch.arange(shape["pos"][-1], dtype=torch.float16, device="cuda").view(*shape["pos"])
    baseline_ms = bench_cuda(lambda: baseline_fn(q, k, pos))
    kernel_ms = bench_cuda(lambda: kernel_fn(q, k, pos))
    speedup = baseline_ms / kernel_ms
    results[name] = {
        "baseline_ms": round(baseline_ms, 4),
        "kernel_ms": round(kernel_ms, 4),
        "speedup": round(speedup, 3),
    }
    print(f"{name}: {baseline_ms:.4f} ms -> {kernel_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
