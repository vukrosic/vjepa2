"""Benchmark for fused_argsort_gather kernel."""
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_argsort_gather import kernel_fn, baseline_fn, SHAPES

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
    B, N, total_D = shape["x"]
    x = torch.randn(B, N, total_D, dtype=torch.float16, device="cuda")
    masks = torch.rand(B, N, device="cuda")
    argsort = torch.argsort(masks, dim=1)
    assert x.is_contiguous()
    base_fn = lambda: baseline_fn(x, argsort)
    kern_fn = lambda: kernel_fn(x, argsort)
    base_ms = bench_cuda(base_fn)
    kern_ms = bench_cuda(kern_fn)
    speedup = base_ms / kern_ms
    results[name] = {"baseline_ms": round(base_ms,4), "kernel_ms": round(kern_ms,4), "speedup": round(speedup,3)}
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")
print(f"BENCH_RESULT={json.dumps(results)}")
