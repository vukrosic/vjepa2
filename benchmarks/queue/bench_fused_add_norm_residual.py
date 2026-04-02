"""Benchmark for fused_add_norm_residual kernel."""
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.utils.kernels.fused_add_norm_residual import can_use_kernel, kernel_fn, baseline_fn, SHAPES


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
    D = shape["D"]
    x = torch.randn(shape["x"], dtype=torch.float32, device="cuda")
    b1 = torch.randn(shape["b1"], dtype=torch.float32, device="cuda")
    b2 = torch.randn(shape["b2"], dtype=torch.float32, device="cuda")
    weight = torch.randn(D, dtype=torch.float32, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, device="cuda")
    assert can_use_kernel(x, b1, b2, weight, bias), f"benchmark shape {name} is not using the supported path"
    base_ms = bench_cuda(lambda: baseline_fn(x, b1, b2, weight, bias))
    kern_ms = bench_cuda(lambda: kernel_fn(x, b1, b2, weight, bias))
    speedup = base_ms / kern_ms
    results[name] = {
        "baseline_ms": round(base_ms, 4),
        "kernel_ms": round(kern_ms, 4),
        "speedup": round(speedup, 3),
    }
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
