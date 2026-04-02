"""Benchmark for fused_token_scatter kernel."""
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_token_scatter import kernel_fn, baseline_fn, SHAPES

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
    src = torch.randn(*shape["src"], dtype=torch.float16, device="cuda")
    total_tokens = shape["total_tokens"]
    B, M, D = src.shape
    torch.manual_seed(0)
    indices = torch.randint(0, total_tokens, (B, M), device="cuda")
    base_fn = lambda: baseline_fn(src, indices, total_tokens, D)
    kern_fn = lambda: kernel_fn(src, indices, total_tokens, D)
    base_ms = bench_cuda(base_fn)
    kern_ms = bench_cuda(kern_fn)
    speedup = base_ms / kern_ms
    results[name] = {"baseline_ms": round(base_ms, 4), "kernel_ms": round(kern_ms, 4), "speedup": round(speedup, 3)}
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
