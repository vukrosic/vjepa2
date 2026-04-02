"""Benchmark for fused Sigmoid * Multiply kernel."""
import json
import sys
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_sigmoid_mul import kernel_fn, baseline_fn, SHAPES


def bench_cuda(fn, warmup=30, iters=200):
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
    x_gate = torch.randn(*shape["x"], dtype=torch.float16, device="cuda")
    y_value = torch.randn(*shape["y"], dtype=torch.float16, device="cuda")
    base_ms = bench_cuda(lambda: baseline_fn(x_gate, y_value))
    kern_ms = bench_cuda(lambda: kernel_fn(x_gate, y_value))
    speedup = base_ms / kern_ms
    results[name] = {
        "baseline_ms": round(base_ms, 4),
        "kernel_ms": round(kern_ms, 4),
        "speedup": round(speedup, 3),
    }
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
