"""Benchmark for fused_cross_entropy kernel."""
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_cross_entropy import kernel_fn, baseline_fn, SHAPES

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
    logits = torch.randn(*shape["logits"], dtype=torch.float32, device="cuda")
    labels = torch.randint(0, shape["logits"][1], shape["labels_shape"], device="cuda").long()
    temperature = shape["temperature"]
    base_ms = bench_cuda(lambda: baseline_fn(logits, labels, temperature))
    kern_ms = bench_cuda(lambda: kernel_fn(logits, labels, temperature))
    speedup = base_ms / kern_ms
    results[name] = {"baseline_ms": round(base_ms, 4), "kernel_ms": round(kern_ms, 4), "speedup": round(speedup, 3)}
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
