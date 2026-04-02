"""Benchmark for fused_norm_linear kernel."""
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_norm_linear import kernel_fn, baseline_fn, SHAPES

def bench_cuda(fn, warmup=30, iters=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

results = {}
for name, shape in SHAPES.items():
    x = torch.randn(shape["x"][0], shape["x"][1], shape["D_in"], dtype=torch.float16, device="cuda")
    weight = torch.randn(shape["D_out"], shape["D_in"], dtype=torch.float16, device="cuda")
    bias = torch.randn(shape["D_out"], dtype=torch.float16, device="cuda")
    ln_weight = torch.ones(shape["D_in"], dtype=torch.float16, device="cuda")
    ln_bias = torch.zeros(shape["D_in"], dtype=torch.float16, device="cuda")
    base_ms = bench_cuda(lambda: baseline_fn(x, weight, bias, ln_weight, ln_bias))
    kern_ms = bench_cuda(lambda: kernel_fn(x, weight, bias, ln_weight, ln_bias))
    speedup = base_ms / kern_ms
    results[name] = {"baseline_ms": round(base_ms, 4), "kernel_ms": round(kern_ms, 4), "speedup": round(speedup, 3)}
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
