import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.fused_softmax_cross_entropy_logits import kernel_fn, baseline_fn, SHAPES

def bench(shape_name):
    shape = SHAPES[shape_name]
    logits = torch.randn(*shape["logits"], dtype=torch.float32, device="cuda")
    targets = torch.randint(0, shape["logits"][1], shape["target_shape"], device="cuda")
    temperature = 0.07
    warmup = 30; iters = 200
    for _ in range(warmup): torch.cuda.synchronize(); baseline_fn(logits, targets, temperature)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): baseline_fn(logits, targets, temperature)
    t1.record(); torch.cuda.synchronize()
    t_bas = t0.elapsed_time(t1) / iters
    for _ in range(warmup): torch.cuda.synchronize(); kernel_fn(logits, targets, temperature)
    torch.cuda.synchronize()
    t0.record()
    for _ in range(iters): kernel_fn(logits, targets, temperature)
    t1.record(); torch.cuda.synchronize()
    t_ker = t0.elapsed_time(t1) / iters
    return {"shape": shape_name, "dtype": "float32", "baseline_ms": t_bas, "kernel_ms": t_ker}

if __name__ == "__main__":
    results = []
    for shape_name in SHAPES:
        results.append(bench(shape_name))
    print(f"BENCH_RESULT={json.dumps(results)}")
