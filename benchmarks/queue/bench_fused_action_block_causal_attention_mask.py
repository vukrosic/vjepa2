"""Benchmark for build_action_block_causal_attention_mask."""

from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.utils.kernels.fused_action_block_causal_attention_mask import (
    SHAPES,
    baseline_fn,
    kernel_fn,
)


def bench_cpu(fn, warmup=200, iters=2000):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


results = {}
for name, shape in SHAPES.items():
    baseline = lambda s=shape: baseline_fn(s["T"], s["H"], s["W"], s["add_tokens"])
    optimized = lambda s=shape: kernel_fn(s["T"], s["H"], s["W"], s["add_tokens"])

    baseline_out = baseline()
    optimized_out = optimized()
    assert baseline_out.shape == optimized_out.shape
    assert baseline_out.dtype == optimized_out.dtype
    assert baseline_out.device == optimized_out.device

    baseline_ms = bench_cpu(baseline)
    optimized_ms = bench_cpu(optimized)
    speedup = baseline_ms / optimized_ms if optimized_ms else float("inf")
    results[name] = {
        "baseline_ms": round(baseline_ms, 4),
        "kernel_ms": round(optimized_ms, 4),
        "speedup": round(speedup, 3),
    }
    print(f"{name}: {baseline_ms:.4f} ms -> {optimized_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
