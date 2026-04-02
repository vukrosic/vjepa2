"""Benchmark for fused_swish_bw kernel."""
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_swish_bw import kernel_fn, SHAPES

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
    x = torch.randn(*shape["x"], dtype=torch.float16, device="cuda")
    dy = torch.randn(*shape["x"], dtype=torch.float16, device="cuda")
    kern_ms = bench_cuda(lambda: kernel_fn(x, dy))
    results[name] = {"kernel_ms": round(kern_ms,4)}
    print(f"{name}: {kern_ms:.4f} ms")
print(f"BENCH_RESULT={json.dumps(results)}")
