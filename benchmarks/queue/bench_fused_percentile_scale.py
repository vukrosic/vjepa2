import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_percentile_scale import kernel_fn, baseline_fn, SHAPES

def bench(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    gamma = torch.randn(*shape["gamma"], dtype=dtype, device="cuda")
    warmup = 30; iters = 200
    for _ in range(warmup): torch.cuda.synchronize(); baseline_fn(x, gamma)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): baseline_fn(x, gamma)
    t1.record(); torch.cuda.synchronize()
    t_bas = t0.elapsed_time(t1) / iters
    for _ in range(warmup): torch.cuda.synchronize(); kernel_fn(x, gamma)
    torch.cuda.synchronize()
    t0.record()
    for _ in range(iters): kernel_fn(x, gamma)
    t1.record(); torch.cuda.synchronize()
    t_ker = t0.elapsed_time(t1) / iters
    return {"shape": shape_name, "dtype": str(dtype), "baseline_ms": t_bas, "kernel_ms": t_ker}

if __name__ == "__main__":
    results = []
    for shape_name in SHAPES:
        for dtype in [torch.float16, torch.float32]:
            results.append(bench(shape_name, dtype))
    print(f"BENCH_RESULT={json.dumps(results)}")
