"""Benchmark for fused_gradient_clip kernel."""
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_gradient_clip import kernel_fn, baseline_fn, SHAPES

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
max_norm = 1.0

for name, shape in SHAPES.items():
    grad = torch.randn(*shape["grad"], dtype=torch.float32, device="cuda") * 0.01
    grads_b = [g.clone() for g in [grad]]
    grads_k = [g.clone() for g in [grad]]
    base_ms = bench_cuda(lambda: baseline_fn(grads_b, max_norm))
    kern_ms = bench_cuda(lambda: kernel_fn(grads_k, max_norm))
    speedup = base_ms / kern_ms
    results[name] = {"baseline_ms": round(base_ms, 4), "kernel_ms": round(kern_ms, 4), "speedup": round(speedup, 3)}
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
