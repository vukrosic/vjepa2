# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark for Triton mask kernels against baseline apply_masks."""

import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.masks.utils import apply_masks as baseline_apply_masks
from src.models.utils.mask_kernels import triton_apply_masks, triton_apply_masks_single


def benchmark_apply_masks_single_2d():
    """Benchmark single 2D mask application."""
    B, N, D = 8, 1024, 256
    K = 256
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    mask = torch.randint(0, N, (B, K), device="cuda", dtype=torch.long)

    def baseline():
        return baseline_apply_masks(x, [mask], concat=True)

    def optimized():
        return triton_apply_masks_single(x, mask)

    torch.cuda.synchronize()
    for _ in range(10):
        baseline()
        optimized()
    torch.cuda.synchronize()

    baseline_out = baseline()
    optimized_out = optimized()
    max_abs_diff = (baseline_out - optimized_out).abs().max().item()

    def timed(fn, warmup=10, iters=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0 / iters

    baseline_ms = timed(baseline)
    optimized_ms = timed(optimized)

    return {
        "name": "apply_masks_single_2d",
        "baseline_ms": baseline_ms,
        "optimized_ms": optimized_ms,
        "speedup_pct": ((baseline_ms - optimized_ms) / baseline_ms) * 100.0,
        "max_abs_diff": max_abs_diff,
    }


def benchmark_apply_masks_single_1d():
    """Benchmark single 1D mask application."""
    B, N, D = 8, 1024, 256
    K = 256
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    mask = torch.randint(0, N, (K,), device="cuda", dtype=torch.long)

    def baseline():
        return baseline_apply_masks(x, [mask], concat=True)

    def optimized():
        return triton_apply_masks_single(x, mask)

    torch.cuda.synchronize()
    for _ in range(10):
        baseline()
        optimized()
    torch.cuda.synchronize()

    baseline_out = baseline()
    optimized_out = optimized()
    max_abs_diff = (baseline_out - optimized_out).abs().max().item()

    def timed(fn, warmup=10, iters=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0 / iters

    baseline_ms = timed(baseline)
    optimized_ms = timed(optimized)

    return {
        "name": "apply_masks_single_1d",
        "baseline_ms": baseline_ms,
        "optimized_ms": optimized_ms,
        "speedup_pct": ((baseline_ms - optimized_ms) / baseline_ms) * 100.0,
        "max_abs_diff": max_abs_diff,
    }


def benchmark_apply_masks_multi_2d():
    """Benchmark multiple 2D masks application (the common case)."""
    B, N, D = 8, 1024, 256
    K = 256
    M = 4
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    masks = [torch.randint(0, N, (B, K), device="cuda", dtype=torch.long) for _ in range(M)]

    def baseline():
        return baseline_apply_masks(x, masks)

    def optimized():
        return triton_apply_masks(x, masks)

    torch.cuda.synchronize()
    for _ in range(10):
        baseline()
        optimized()
    torch.cuda.synchronize()

    baseline_out = baseline()
    optimized_out = optimized()
    max_abs_diff = (baseline_out - optimized_out).abs().max().item()

    def timed(fn, warmup=10, iters=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0 / iters

    baseline_ms = timed(baseline)
    optimized_ms = timed(optimized)

    return {
        "name": "apply_masks_multi_2d",
        "baseline_ms": baseline_ms,
        "optimized_ms": optimized_ms,
        "speedup_pct": ((baseline_ms - optimized_ms) / baseline_ms) * 100.0,
        "max_abs_diff": max_abs_diff,
    }


def benchmark_apply_masks_multi_1d():
    """Benchmark multiple 1D masks application."""
    B, N, D = 8, 1024, 256
    K = 256
    M = 4
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    masks = [torch.randint(0, N, (K,), device="cuda", dtype=torch.long) for _ in range(M)]

    def baseline():
        return baseline_apply_masks(x, masks)

    def optimized():
        return triton_apply_masks(x, masks)

    torch.cuda.synchronize()
    for _ in range(10):
        baseline()
        optimized()
    torch.cuda.synchronize()

    baseline_out = baseline()
    optimized_out = optimized()
    max_abs_diff = (baseline_out - optimized_out).abs().max().item()

    def timed(fn, warmup=10, iters=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0 / iters

    baseline_ms = timed(baseline)
    optimized_ms = timed(optimized)

    return {
        "name": "apply_masks_multi_1d",
        "baseline_ms": baseline_ms,
        "optimized_ms": optimized_ms,
        "speedup_pct": ((baseline_ms - optimized_ms) / baseline_ms) * 100.0,
        "max_abs_diff": max_abs_diff,
    }


def benchmark_apply_masks_large():
    """Benchmark with larger dimensions."""
    B, N, D = 16, 2048, 512
    K = 512
    M = 4
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    masks = [torch.randint(0, N, (B, K), device="cuda", dtype=torch.long) for _ in range(M)]

    def baseline():
        return baseline_apply_masks(x, masks)

    def optimized():
        return triton_apply_masks(x, masks)

    torch.cuda.synchronize()
    for _ in range(5):
        baseline()
        optimized()
    torch.cuda.synchronize()

    baseline_out = baseline()
    optimized_out = optimized()
    max_abs_diff = (baseline_out - optimized_out).abs().max().item()

    def timed(fn, warmup=5, iters=30):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0 / iters

    baseline_ms = timed(baseline)
    optimized_ms = timed(optimized)

    return {
        "name": "apply_masks_large",
        "baseline_ms": baseline_ms,
        "optimized_ms": optimized_ms,
        "speedup_pct": ((baseline_ms - optimized_ms) / baseline_ms) * 100.0,
        "max_abs_diff": max_abs_diff,
    }


def benchmark_apply_masks_small():
    """Benchmark with smaller dimensions."""
    B, N, D = 4, 512, 128
    K = 128
    M = 4
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    masks = [torch.randint(0, N, (B, K), device="cuda", dtype=torch.long) for _ in range(M)]

    def baseline():
        return baseline_apply_masks(x, masks)

    def optimized():
        return triton_apply_masks(x, masks)

    torch.cuda.synchronize()
    for _ in range(10):
        baseline()
        optimized()
    torch.cuda.synchronize()

    baseline_out = baseline()
    optimized_out = optimized()
    max_abs_diff = (baseline_out - optimized_out).abs().max().item()

    def timed(fn, warmup=10, iters=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0 / iters

    baseline_ms = timed(baseline)
    optimized_ms = timed(optimized)

    return {
        "name": "apply_masks_small",
        "baseline_ms": baseline_ms,
        "optimized_ms": optimized_ms,
        "speedup_pct": ((baseline_ms - optimized_ms) / baseline_ms) * 100.0,
        "max_abs_diff": max_abs_diff,
    }


def bench(name, baseline_fn, optimized_fn, warmup=10, iters=50):
    """Generic benchmark function following kernel_speedups.py methodology."""
    for _ in range(warmup):
        baseline_fn()
        optimized_fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        baseline_fn()
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - start) * 1000.0 / iters

    start = time.perf_counter()
    for _ in range(iters):
        optimized_fn()
    torch.cuda.synchronize()
    optimized_ms = (time.perf_counter() - start) * 1000.0 / iters

    with torch.no_grad():
        baseline_out = baseline_fn()
        optimized_out = optimized_fn()
        max_abs_diff = (baseline_out - optimized_out).abs().max().item()

    return {
        "name": name,
        "baseline_ms": baseline_ms,
        "optimized_ms": optimized_ms,
        "speedup_pct": ((baseline_ms - optimized_ms) / baseline_ms) * 100.0,
        "max_abs_diff": max_abs_diff,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    results = []

    # Run individual benchmarks
    results.append(benchmark_apply_masks_single_2d())
    results.append(benchmark_apply_masks_single_1d())
    results.append(benchmark_apply_masks_multi_2d())
    results.append(benchmark_apply_masks_multi_1d())
    results.append(benchmark_apply_masks_large())
    results.append(benchmark_apply_masks_small())

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
