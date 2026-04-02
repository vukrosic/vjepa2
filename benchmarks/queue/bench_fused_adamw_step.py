"""Benchmark for fused_adamw_step kernel."""
import json, sys, pathlib, torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_adamw_step import kernel_fn, baseline_fn, SHAPES


def bench_cuda(fn, warmup=30, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


results = {}
lr, beta1, beta2, eps, wd, step = 1e-3, 0.9, 0.999, 1e-8, 0.01, 100

for name, shape in SHAPES.items():
    param_shape = shape["param"]
    param_b = torch.randn(*param_shape, dtype=torch.float32, device="cuda")
    param_k = param_b.clone()
    grad = torch.randn(*param_shape, dtype=torch.float32, device="cuda") * 0.01
    m_b = torch.zeros_like(param_b)
    v_b = torch.zeros_like(param_b)
    m_k = torch.zeros_like(param_b)
    v_k = torch.zeros_like(param_b)

    base_ms = bench_cuda(lambda: baseline_fn(param_b, grad, m_b, v_b, lr, beta1, beta2, eps, wd, step))
    kern_ms = bench_cuda(lambda: kernel_fn(param_k, grad, m_k, v_k, lr, beta1, beta2, eps, wd, step))
    speedup = base_ms / kern_ms
    results[name] = {
        "baseline_ms": round(base_ms, 4),
        "kernel_ms": round(kern_ms, 4),
        "speedup": round(speedup, 3),
    }
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")

print(f"BENCH_RESULT={json.dumps(results)}")
