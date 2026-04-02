"""Benchmark for fused_momentum_teacher kernel."""
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_momentum_teacher import kernel_fn, baseline_fn, SHAPES

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
    teacher = torch.randn(shape["teacher"], dtype=torch.float32, device="cuda")
    student = torch.randn(shape["student"], dtype=torch.float32, device="cuda")
    momentum = 0.996
    temperature = 0.07
    base_ms = bench_cuda(lambda: baseline_fn(teacher.clone(), student, momentum, temperature))
    kern_ms = bench_cuda(lambda: kernel_fn(teacher, student, momentum, temperature))
    speedup = base_ms / kern_ms
    results[name] = {"baseline_ms": round(base_ms,4), "kernel_ms": round(kern_ms,4), "speedup": round(speedup,3)}
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")
print(f"BENCH_RESULT={json.dumps(results)}")
