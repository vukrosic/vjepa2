"""Benchmark for fused_layernorm_rmsnorm kernel."""
import json
import sys
import pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_layernorm_rmsnorm import kernel_fn, baseline_fn, SHAPES


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
    x = torch.randn(shape["x"], dtype=torch.float32, device="cuda")
    ln_D = shape["LN_D"]
    rms_D = shape["RMS_D"]
    ln_weight = torch.ones(ln_D, dtype=torch.float32, device="cuda")
    ln_bias = torch.zeros(ln_D, dtype=torch.float32, device="cuda")
    rms_weight = torch.ones(rms_D, dtype=torch.float32, device="cuda")
    base_ms = bench_cuda(lambda: baseline_fn(x, ln_weight, ln_bias, rms_weight, ln_D))
    kern_ms = bench_cuda(lambda: kernel_fn(x, ln_weight, ln_bias, rms_weight, ln_D))
    speedup = base_ms / kern_ms
    results[name] = {"baseline_ms": round(base_ms, 4), "kernel_ms": round(kern_ms, 4), "speedup": round(speedup, 3)}
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")
print(f"BENCH_RESULT={json.dumps(results)}")
