"""Benchmark for fused_qkv_split kernel."""
import json
import sys
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_qkv_split import kernel_fn, baseline_fn, SHAPES


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
    B, N, H, D = shape["B"], shape["N"], shape["H"], shape["D"]
    qkv_linear = torch.randn(*shape["qkv_linear"], dtype=torch.float16, device="cuda")
    base_ms = bench_cuda(lambda: baseline_fn(qkv_linear, B, N, H, D))
    kern_ms = bench_cuda(lambda: kernel_fn(qkv_linear, B, N, H, D))
    speedup = base_ms / kern_ms
    results[name] = {
        "baseline_ms": round(base_ms, 4),
        "kernel_ms": round(kern_ms, 4),
        "speedup": round(speedup, 3),
    }
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
