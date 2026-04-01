# Kernel Measurement Analysis for vjepa2

This document analyzes how kernel performance is measured across the vjepa2 codebase.

---

## 1. Current Measurement Approach

The codebase uses two distinct timing mechanisms:

### 1.1 CUDA Events (Preferred)

Used in `tests/support/kernel_benchmark.py`, `benchmarks/baseline_compare.py`, `benchmarks/app_rope_module_compare.py`, and `benchmarks/app_predictor_compare.py`.

```python
# From tests/support/kernel_benchmark.py lines 51-62
def benchmark_cuda(fn, warmup, iters):
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
```

This is the **correct approach** for GPU timing:
- `torch.cuda.Event` records timestamps on the GPU timeline
- `synchronize()` ensures all prior GPU operations complete before timing starts
- `elapsed_time()` returns milliseconds measured entirely on the GPU

### 1.2 Python `time.perf_counter` (Suboptimal)

Used in `benchmarks/kernel_speedups.py` for some benchmarks.

```python
# From benchmarks/kernel_speedups.py lines 79-87
def timed(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters
```

**Problems:**
- `time.perf_counter()` is a CPU-side timer that includes Python overhead
- `synchronize()` before timing ensures GPU is ready, but the timer itself runs on CPU
- Any Python interpreter overhead (GC, function call overhead) pollutes measurements
- Should use CUDA events instead

### 1.3 Standard `time.perf_counter` for CPU Functions

Used for CPU-only operations like `build_action_block_causal_attention_mask`:

```python
# From tests/support/kernel_benchmark.py lines 41-48
def benchmark_cpu(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters
```

This is appropriate for CPU-only functions.

---

## 2. Warmup and Iteration Strategy

### 2.1 Standard Pattern

Most benchmarks follow this pattern:

```python
# Warmup iterations (discards results)
for _ in range(warmup):
    fn()

# Timed iterations
torch.cuda.synchronize()
start = torch.cuda.Event(...)
for _ in range(iters):
    fn()
end.record()
torch.cuda.synchronize()
return start.elapsed_time(end) / iters
```

### 2.2 Typical Values

| File | Warmup | Timed Iterations |
|------|--------|------------------|
| `kernel_speedups.py` | 10-20 | 40-80 |
| `kernel_benchmark.py` | 10-20 | 20-200 |
| `baseline_compare.py` | 10-20 | 50-200 |
| `app_rope_module_compare.py` | 20-40 | 100-400 |
| `app_predictor_compare.py` | 20 | 100 |

### 2.3 Backward Pass Timing

For training backward passes, `kernel_benchmark.py` has special handling:

```python
# From tests/support/kernel_benchmark.py lines 65-78
def benchmark_cuda_backward(fn, warmup, iters):
    for _ in range(warmup):
        x = fn()
        x.square().mean().backward()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        x = fn()
        x.square().mean().backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters
```

Note: This uses a fixed `x.square().mean().backward()` as the loss, not arbitrary user-defined losses.

### 2.4 Multi-Sample Pattern (Most Sophisticated)

`tests/support/kernel_benchmark.py` supports collecting multiple samples:

```python
# From tests/support/kernel_benchmark.py lines 111-124
samples = []
for _ in range(repeats):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timed_iters):
        run_attention_policy(...)
    end.record()
    torch.cuda.synchronize()
    samples.append(start.elapsed_time(end) / timed_iters)
```

This enables statistical aggregation (median, mean, min).

---

## 3. Variance Handling

### 3.1 Single-Point Estimate (Most Common)

Most benchmark files return only a single timing value (average over timed iterations):

```python
return start.elapsed_time(end) / iters
```

No statistical distribution is captured.

### 3.2 Statistical Aggregation (Best Practice)

`tests/support/kernel_benchmark.py` captures and reports multiple statistics:

```python
# From tests/support/kernel_benchmark.py lines 126-137
return KernelResult(
    ...
    median_ms=median(samples),
    mean_ms=mean(samples),
    min_ms=min(samples),
    samples_ms=tuple(samples),
)
```

The `KernelResult` dataclass:

```python
# From tests/support/kernel_benchmark.py lines 19-30
@dataclass(frozen=True)
class KernelResult:
    case: str
    policy: str
    masked: bool
    dtype: str
    shape: tuple[int, int, int, int]
    backend_order: tuple[str, ...]
    median_ms: float
    mean_ms: float
    min_ms: float
    samples_ms: tuple[float, ...]
```

### 3.3 Speedup Calculation

All files calculate speedup consistently:

```python
speedup_pct = ((baseline_ms - optimized_ms) / baseline_ms) * 100.0
speedup_x = baseline_ms / optimized_ms  # Used in kernel_benchmark.py
```

---

## 4. Parity Checking

### 4.1 Forward Pass Parity

```python
torch.testing.assert_close(optimized_output, baseline_output, atol=tol, rtol=tol)
```

Typical tolerances:
- `atol=5e-3, rtol=5e-3` for float16/bfloat16 RoPE operations
- `atol=2e-3, rtol=2e-3` for attention modules
- `atol=0, rtol=0` for exact parity (e.g., `rotate_query_key_pair`)

### 4.2 Training Parity (Forward + Backward)

Training benchmarks verify both forward outputs and gradients:

```python
# From tests/support/kernel_benchmark.py lines 238-255
ref_q = q.detach().clone().requires_grad_(True)
ref_k = k.detach().clone().requires_grad_(True)
baseline_q, baseline_k = baseline_modules.rotate_query_key_pair(ref_q, ref_k, pos)
(baseline_q.square().mean() + baseline_k.square().mean()).backward()
ref_q_grad = ref_q.grad.detach().clone()
ref_k_grad = ref_k.grad.detach().clone()

opt_q = q.detach().clone().requires_grad_(True)
opt_k = k.detach().clone().requires_grad_(True)
optimized_q, optimized_k = rotate_query_key_pair(opt_q, opt_k, pos)
(optimized_q.square().mean() + optimized_k.square().mean()).backward()
opt_q_grad = opt_q.grad.detach().clone()
opt_k_grad = opt_k.grad.detach().clone()

torch.testing.assert_close(optimized_q, baseline_q, atol=5e-3, rtol=5e-3)
torch.testing.assert_close(optimized_k, baseline_k, atol=5e-3, rtol=5e-3)
torch.testing.assert_close(opt_q_grad, ref_q_grad, atol=5e-3, rtol=5e-3)
torch.testing.assert_close(opt_k_grad, ref_k_grad, atol=5e-3, rtol=5e-3)
```

### 4.3 Tolerance Selection

```python
# From tests/support/kernel_benchmark.py lines 105-109
if check_parity:
    baseline = run_attention_policy("default_auto", q, k, v, ...)
    candidate = run_attention_policy(policy, q, k, v, ...)
    tol = 2e-3 if case.dtype in (torch.float16, torch.bfloat16) else 1e-5
    torch.testing.assert_close(candidate, baseline, atol=tol, rtol=tol)
```

---

## 5. Limitations

### 5.1 No Memory Measurement

No benchmark captures memory usage:
- No peak GPU memory per kernel
- No memory bandwidth calculation
- No L2 cache hit rates

### 5.2 No FLOP Estimation

No measurement of actual compute:
- No FLOP/s throughput calculation
- No roofline model comparison
- No theoretical vs actual performance ratio

### 5.3 No Kernel-Level Breakdown

Timing captures end-to-end function time:
- No visibility into individual CUDA kernel launches
- Cannot identify bottleneck kernels
- No profiler integration (nsys, PyTorch profiler)

### 5.4 Using `time.perf_counter` for CUDA

In `benchmarks/kernel_speedups.py`:

```python
start = time.perf_counter()  # CPU timer
for _ in range(iters):
    fn()
torch.cuda.synchronize()
return (time.perf_counter() - start) * 1000.0 / iters
```

**Issue:** The loop executes on CPU, so CPU-GPU synchronization overhead and Python call overhead are included. For short-running kernels, this can cause significant measurement error.

### 5.5 No Cold-Start vs Steady-State Distinction

All benchmarks assume steady-state timing:
- No measurement of kernel compilation time (CUDA JIT)
- No separation of first-run vs subsequent runs
- Triton/TorchScript compilation not measured

### 5.6 Inconsistent Aggregation

Only `tests/support/kernel_benchmark.py` captures distribution statistics. Other files return single averages, making it impossible to:
- Detect outlier runs
- Calculate confidence intervals
- Identify thermal throttling effects

### 5.7 No GPU Frequency Measurement

GPU clock speed affects performance:
- No measurement of sustained vs boosted clocks
- No detection of throttling
- Results may vary based on power/temperature state

### 5.8 Baseline Loading from git HEAD

Some benchmarks load baseline code from git:

```python
# From tests/support/kernel_benchmark.py lines 29-38
def load_module_from_head(repo_root, git_path, module_name):
    content = subprocess.check_output(["git", "show", f"HEAD:{git_path}"], cwd=repo_root, text=True)
    # ... load into temp file
```

This ensures a clean baseline comparison but:
- Requires git repository to be intact
- Does not work with exported/deployed code
- Assumes HEAD is the correct baseline

---

## 6. Recommendations

### 6.1 Standardize on CUDA Events

Replace all `time.perf_counter()` usage with CUDA events:

```python
def benchmark_cuda(fn, warmup, iters):
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
```

### 6.2 Collect Multiple Samples with Statistical Reporting

Adopt the `KernelResult` pattern from `kernel_benchmark.py`:

```python
@dataclass(frozen=True)
class KernelResult:
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    samples_ms: tuple[float, ...]
```

### 6.3 Add Memory Measurement

```python
torch.cuda.reset_peak_memory_stats()
# ... run kernel ...
peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
```

### 6.4 Add FLOP Estimation

For matrix operations, theoretical FLOPs:
- GEMM: `2 * M * N * K` for MxK @ KxN
- Attention: `2 * B * H * N * N * D` for attention scores + 2 * B * H * N * N * D for weighted sum

### 6.5 Add Kernel Profiling Integration

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    fn()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 6.6 Separate Warmup from Timing

Distinguish cold-start from steady-state:

```python
# Cold-start measurement
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
fn()
end.record()
torch.cuda.synchronize()
cold_start_ms = start.elapsed_time(end)

# Steady-state measurement (after warmup)
for _ in range(warmup):
    fn()
# ... then timed iterations ...
```

### 6.7 Consider Using `torch.utils.benchmark`

For more robust measurements:

```python
import torch.utils.benchmark as benchmark

def benchmark_fn():
    return fn()

timer = benchmark.Timer(
    stmt='benchmark_fn()',
    globals={'benchmark_fn': benchmark_fn},
    num_threads=1,
)
print(timer.timeit(number=100).mean)
```

---

## Summary

| Aspect | Status |
|--------|--------|
| CUDA Event Usage | Partial (some files use perf_counter) |
| Warmup Strategy | Basic (fixed warmup iterations) |
| Variance Handling | Poor (single-point estimates in most files) |
| Statistical Aggregation | Only in `kernel_benchmark.py` |
| Memory Measurement | Not implemented |
| FLOP Estimation | Not implemented |
| Kernel-Level Breakdown | Not implemented |
| Cold-start vs Steady-state | Not distinguished |

The most robust measurement infrastructure is in `tests/support/kernel_benchmark.py`, which should serve as the template for future benchmark development.
