"""Benchmark for fused_softmax_rope kernel."""
import json
import sys
import pathlib
import torch
import math

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_softmax_rope import kernel_fn, baseline_fn, SHAPES


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
    T, D, H = shape["T"], shape["D"], shape["H"]
    B = 2
    q = torch.randn(B, H, T, D, dtype=torch.float32, device="cuda")
    k = torch.randn(B, H, T, D, dtype=torch.float32, device="cuda")
    D_half = D // 2
    theta = 10000.0
    positions = torch.arange(D_half, dtype=torch.float32, device="cuda")
    freqs = 1.0 / (theta ** (positions / D_half))
    angles = positions.unsqueeze(0) * freqs.unsqueeze(1)
    cos_table = angles.cos()
    sin_table = angles.sin()
    base_ms = bench_cuda(lambda: baseline_fn(q, k, cos_table, sin_table))
    kern_ms = bench_cuda(lambda: kernel_fn(q, k, cos_table, sin_table))
    speedup = base_ms / kern_ms
    results[name] = {"baseline_ms": round(base_ms, 4), "kernel_ms": round(kern_ms, 4), "speedup": round(speedup, 3)}
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")
print(f"BENCH_RESULT={json.dumps(results)}")
