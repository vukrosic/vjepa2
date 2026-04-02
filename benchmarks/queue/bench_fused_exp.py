"""Benchmark for fused_exp kernel."""
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_exp import kernel_fn, baseline_fn, SHAPES

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
    if "x1" in shape and "x2" in shape:
        x = torch.randn(*shape["x1"], dtype=torch.float16, device="cuda")
        x2 = torch.randn(*shape["x2"], dtype=torch.float16, device="cuda")
        base_ms = bench_cuda(lambda: baseline_fn(x, x2))
        kern_ms = bench_cuda(lambda: kernel_fn(x, x2))
    elif "pred" in shape and "target" in shape:
        pred = torch.randn(*shape["pred"], dtype=torch.float16, device="cuda")
        target = torch.randn(*shape["target"], dtype=torch.float16, device="cuda")
        base_ms = bench_cuda(lambda: baseline_fn(pred, target))
        kern_ms = bench_cuda(lambda: kernel_fn(pred, target))
    elif "x" in shape and "D" in shape:
        x = torch.randn(*shape["x"], dtype=torch.float16, device="cuda")
        w = torch.randn(shape["D"], dtype=torch.float16, device="cuda")
        b = torch.randn(shape["D"], dtype=torch.float16, device="cuda")
        base_ms = bench_cuda(lambda: baseline_fn(x, w, b))
        kern_ms = bench_cuda(lambda: kernel_fn(x, w, b))
    else:
        x = torch.randn(*shape["x"], dtype=torch.float16, device="cuda")
        base_ms = bench_cuda(lambda: baseline_fn(x))
        kern_ms = bench_cuda(lambda: kernel_fn(x))
    speedup = base_ms / kern_ms
    results[name_shape] = {"baseline_ms": round(base_ms,4), "kernel_ms": round(kern_ms,4), "speedup": round(speedup,3)}
    print(f"{name_shape}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")
print(f"BENCH_RESULT={json.dumps(results)}")
