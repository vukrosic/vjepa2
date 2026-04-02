"""Benchmark for fused_ema_update kernel."""
import json
import sys
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_ema_update import kernel_fn, baseline_fn, SHAPES

MOMENTUM = 0.996


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
    # Re-allocate fresh tensors per iteration inside lambda to avoid measuring
    # compounding in-place effects across benchmark iterations. We allocate once
    # and reset in the lambda so timing stays representative.
    target_base = torch.randn(*shape["target"], dtype=torch.float16, device="cuda")
    student = torch.randn(*shape["student"], dtype=torch.float16, device="cuda")
    target_kern = target_base.clone()

    base_ms = bench_cuda(lambda: baseline_fn(target_base, student, MOMENTUM))
    kern_ms = bench_cuda(lambda: kernel_fn(target_kern, student, MOMENTUM))
    speedup = base_ms / kern_ms
    results[name] = {
        "baseline_ms": round(base_ms, 4),
        "kernel_ms": round(kern_ms, 4),
        "speedup": round(speedup, 3),
    }
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
