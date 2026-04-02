"""Benchmark for fused_dropout_residual_norm kernel."""
import json
import sys
import pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_dropout_residual_norm import kernel_fn, baseline_fn, SHAPES


def bench_cuda(fn, warmup=30, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


results = {}
for name, shape in SHAPES.items():
    x = torch.randn(*shape["x"], dtype=torch.float32, device="cuda")
    residual = torch.randn(*shape["x"], dtype=torch.float32, device="cuda")
    D = shape["D"]
    weight = torch.ones(D, dtype=torch.float32, device="cuda")
    bias = torch.zeros(D, dtype=torch.float32, device="cuda")
    # Benchmark with training=False (no dropout) for deterministic comparison
    base_ms = bench_cuda(lambda: baseline_fn(x, residual, weight, bias, p=0.0, training=False))
    kern_ms = bench_cuda(lambda: kernel_fn(x, residual, weight, bias, p=0.0, training=False, seed=42))
    speedup = base_ms / kern_ms
    results[name] = {"baseline_ms": round(base_ms, 4), "kernel_ms": round(kern_ms, 4), "speedup": round(speedup, 3)}
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")
print(f"BENCH_RESULT={json.dumps(results)}")
