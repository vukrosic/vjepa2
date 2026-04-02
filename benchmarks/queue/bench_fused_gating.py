"""Benchmark for fused_gating kernel."""
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_gating import kernel_fn, baseline_fn, SHAPES

def bench_cuda(fn, warmup=30, iters=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

results = {}
for name_shape, shape in SHAPES.items():
    gate = torch.randn(*shape["gate"], dtype=torch.float16, device="cuda")
    input = torch.randn(*shape["input"], dtype=torch.float16, device="cuda")
    base_ms = bench_cuda(lambda: baseline_fn(gate, input))
    kern_ms = bench_cuda(lambda: kernel_fn(gate, input))
    speedup = base_ms / kern_ms
    results[name_shape] = {"baseline_ms": round(base_ms,4), "kernel_ms": round(kern_ms,4), "speedup": round(speedup,3)}
    print(f"{name_shape}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")
print(f"BENCH_RESULT={json.dumps(results)}")
