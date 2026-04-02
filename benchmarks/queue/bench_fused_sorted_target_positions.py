"""Benchmark for sorted target position helper."""

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.utils.kernels.fused_sorted_target_positions import SHAPES, baseline_fn, kernel_fn


def make_masks(shape):
    masks_x = torch.sort(
        torch.randint(0, shape["max_index"], shape["masks_x"], device="cuda", dtype=torch.int64),
        dim=1,
    )[0]
    masks_y = torch.sort(
        torch.randint(0, shape["max_index"], shape["masks_y"], device="cuda", dtype=torch.int64),
        dim=1,
    )[0]
    return masks_x, masks_y


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
    masks_x, masks_y = make_masks(shape)
    baseline = lambda: baseline_fn(masks_x, masks_y)
    optimized = lambda: kernel_fn(masks_x, masks_y)
    torch.testing.assert_close(optimized(), baseline())
    base_ms = bench_cuda(baseline)
    opt_ms = bench_cuda(optimized)
    speedup = base_ms / opt_ms
    results[name] = {
        "baseline_ms": round(base_ms, 4),
        "kernel_ms": round(opt_ms, 4),
        "speedup": round(speedup, 3),
    }
    print(f"{name}: {base_ms:.4f} ms -> {opt_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
