import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_momentum_teacher import kernel_fn, baseline_fn, SHAPES

def bench(shape_name, dtype):
    shape = SHAPES[shape_name]
    teacher = torch.randn(*shape["teacher"], dtype=dtype, device="cuda")
    student = torch.randn(*shape["student"], dtype=dtype, device="cuda")
    momentum = 0.996
    temperature = 0.07
    warmup = 30; iters = 200
    for _ in range(warmup): torch.cuda.synchronize(); baseline_fn(teacher.clone(), student, momentum, temperature)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): baseline_fn(teacher.clone(), student, momentum, temperature)
    t1.record(); torch.cuda.synchronize()
    t_bas = t0.elapsed_time(t1) / iters
    for _ in range(warmup): torch.cuda.synchronize(); kernel_fn(teacher, student, momentum, temperature)
    torch.cuda.synchronize()
    t0.record()
    for _ in range(iters): kernel_fn(teacher, student, momentum, temperature)
    t1.record(); torch.cuda.synchronize()
    t_ker = t0.elapsed_time(t1) / iters
    return {"shape": shape_name, "dtype": str(dtype), "baseline_ms": t_bas, "kernel_ms": t_ker}

if __name__ == "__main__":
    results = []
    for shape_name in SHAPES:
        for dtype in [torch.float16, torch.float32]:
            results.append(bench(shape_name, dtype))
    print(f"BENCH_RESULT={json.dumps(results)}")
