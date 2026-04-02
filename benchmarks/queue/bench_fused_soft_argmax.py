"""Benchmark for fused_soft_argmax kernel."""
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_soft_argmax import kernel_fn, baseline_fn, SHAPES

def bench_cuda(fn, warmup=30, iters=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

results = {}
for name, shape in SHAPES.items():
    x = torch.randn(shape["x"], dtype=torch.float16, device="cuda")
    temp = shape.get("temp", 1.0)
    base_ms = bench_cuda(lambda: baseline_fn(x, temp))
    kern_ms = bench_cuda(lambda: kernel_fn(x, temp))
    speedup = base_ms / kern_ms
    results[name] = {"baseline_ms": round(base_ms,4), "kernel_ms": round(kern_ms,4), "speedup": round(speedup,3)}
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")
print(f"BENCH_RESULT={json.dumps(results)}")
