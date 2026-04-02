"""Benchmark for fused_apply_masks_source."""

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.utils.kernels.fused_apply_masks_source import SHAPES, baseline_fn, kernel_fn


def _make_masks(shape, device):
    if shape["kind"] == "1d":
        return [
            torch.randint(0, shape["x"][1], (shape["mask_len"],), device=device, dtype=torch.long)
            for _ in range(shape["mask_count"])
        ]
    bsz, k = shape["mask_shape"]
    return [
        torch.randint(0, shape["x"][1], (bsz, k), device=device, dtype=torch.long)
        for _ in range(shape["mask_count"])
    ]


def bench_cuda(fn, warmup=15, iters=60):
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
    x = torch.randn(*shape["x"], dtype=torch.float16, device="cuda")
    masks = _make_masks(shape, "cuda")
    baseline_ms = bench_cuda(lambda: baseline_fn(x, masks))
    kernel_ms = bench_cuda(lambda: kernel_fn(x, masks))
    speedup = baseline_ms / kernel_ms
    results[name] = {
        "baseline_ms": round(baseline_ms, 4),
        "kernel_ms": round(kernel_ms, 4),
        "speedup": round(speedup, 3),
    }
    print(f"{name}: {baseline_ms:.4f} ms -> {kernel_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
