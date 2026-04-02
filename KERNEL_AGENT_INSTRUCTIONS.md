# V-JEPA 2 Kernel Factory — Agent Instructions

You are an optimization agent for Meta's V-JEPA 2 video foundation model. Your job is to generate Triton kernels and PyTorch optimizations, add them to the test queue, and run them on GPU. You work in a tight loop: write kernel, write parity test, write benchmark, submit to queue, move on.

## Setup

```bash
cd /root/vjepa2
# Sync to GPU (update host/port from CLAUDE.md or ask user):
rsync -avz --exclude '.git' -e 'ssh -i ~/.ssh/vast_ai_ed25519 -p PORT' /root/vjepa2/ root@HOST:/workspace/vjepa2/
# Run queue on GPU:
ssh -i ~/.ssh/vast_ai_ed25519 -p PORT root@HOST 'cd /workspace/vjepa2 && python queue_runner.py'
```

## The Loop

For each optimization idea:

1. **Write the kernel** in `src/models/utils/kernels/` (new file per kernel)
2. **Write the parity test** in `tests/queue/test_KERNELNAME.py`
3. **Write the benchmark** in `benchmarks/queue/bench_KERNELNAME.py`
4. **Register it** by adding an entry to `queue/pending.jsonl`
5. **Move to the next kernel immediately** — don't wait for results

The queue runner on GPU processes entries sequentially: parity test first, benchmark second. Results go to `queue/results/`.

## File Conventions

### Kernel files: `src/models/utils/kernels/KERNELNAME.py`

Every kernel file must export:
- `kernel_fn(*args)` — the optimized version
- `baseline_fn(*args)` — the original PyTorch version (copy from source)
- `can_use_kernel(*args) -> bool` — guard for when the kernel applies
- `SHAPES` — dict of realistic test shapes

```python
"""Fused GELU activation kernel."""
import torch
import triton
import triton.language as tl

# --- BASELINE (exact copy from source) ---
def baseline_fn(x):
    return torch.nn.functional.gelu(x)

# --- KERNEL ---
@triton.jit
def _gelu_fwd_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    # GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    y = 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    tl.store(Y + offs, y, mask=mask)

def kernel_fn(x):
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _gelu_fwd_kernel[grid](x, y, N, BLOCK=BLOCK, num_warps=4)
    return y

def can_use_kernel(x):
    return x.is_cuda and x.is_contiguous()

# Realistic shapes from V-JEPA 2 forward pass
SHAPES = {
    "small": {"x": (2, 256, 384)},     # small ViT
    "medium": {"x": (2, 1024, 768)},   # ViT-L
    "large": {"x": (2, 4096, 1024)},   # ViT-H
}
```

### Parity test: `tests/queue/test_KERNELNAME.py`

```python
"""Parity test for KERNELNAME kernel."""
import torch
import pytest
from src.models.utils.kernels.KERNELNAME import kernel_fn, baseline_fn, SHAPES

ATOL_FP16 = 5e-3
ATOL_FP32 = 1e-5

@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_parity(shape_name, dtype):
    shape = SHAPES[shape_name]
    x = torch.randn(*shape["x"], dtype=dtype, device="cuda")
    expected = baseline_fn(x)
    actual = kernel_fn(x)
    atol = ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)

@pytest.mark.parametrize("shape_name", list(SHAPES.keys()))
def test_backward_parity(shape_name):
    """Only if kernel supports backward."""
    shape = SHAPES[shape_name]
    x1 = torch.randn(*shape["x"], dtype=torch.float32, device="cuda", requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    out1 = baseline_fn(x1)
    out2 = kernel_fn(x2)
    grad = torch.randn_like(out1)
    out1.backward(grad)
    out2.backward(grad)
    torch.testing.assert_close(x2.grad, x1.grad, atol=1e-4, rtol=1e-4)
```

### Benchmark: `benchmarks/queue/bench_KERNELNAME.py`

```python
"""Benchmark for KERNELNAME kernel."""
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.utils.kernels.KERNELNAME import kernel_fn, baseline_fn, SHAPES

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
for name, shape in SHAPES.items():
    x = torch.randn(*shape["x"], dtype=torch.float16, device="cuda")
    base_ms = bench_cuda(lambda: baseline_fn(x))
    kern_ms = bench_cuda(lambda: kernel_fn(x))
    speedup = base_ms / kern_ms
    results[name] = {"baseline_ms": round(base_ms, 4), "kernel_ms": round(kern_ms, 4), "speedup": round(speedup, 3)}
    print(f"{name}: {base_ms:.4f} ms -> {kern_ms:.4f} ms ({speedup:.2f}x)")

# Machine-readable output
print(f"BENCH_RESULT={json.dumps(results)}")
```

### Queue entry: append to `queue/pending.jsonl`

```json
{"id": "gelu_fused_001", "kernel": "gelu_fused", "test": "tests/queue/test_gelu_fused.py", "bench": "benchmarks/queue/bench_gelu_fused.py", "target_file": "src/models/utils/modules.py", "target_lines": "153-158", "description": "Fused GELU activation replacing F.gelu in MLP.forward"}
```

## Queue Runner

The queue runner (`queue_runner.py`) is already in the repo. It:
1. Reads `queue/pending.jsonl` line by line
2. Runs `pytest {test_file} -x -q` — if any test fails, marks FAILED, moves on
3. If parity passes, runs `python {bench_file}` — captures BENCH_RESULT JSON
4. Writes result to `queue/results/{id}.json`
5. Moves the entry from pending to `queue/completed.jsonl`

## What To Optimize — Kernel Ideas

Generate kernels for ALL of these. Each is a separate kernel file + test + benchmark + queue entry.

### TIER 1 — High Impact (do these first)

#### 1. Fused LayerNorm + Residual Add
**Source:** `src/models/utils/modules.py:647-663` (Block.forward)
**Pattern:** `x = x + drop_path(attn(norm1(x)))` then `x = x + drop_path(mlp(norm2(x)))`
**Kernel idea:** Fuse `residual_add + layer_norm` into one pass. The kernel already exists in `src/models/utils/fused_layernorm_residual_kernel.py` — adapt it into the queue format and benchmark it against the unfused path.
**Frequency:** 2x per block x 12-40 blocks = 24-80 calls per forward pass.

#### 2. Fused SwiGLU MLP
**Source:** `src/models/utils/modules.py:178-182`
**Pattern:** `F.silu(fc1(x)) * fc2(x)` then `fc3(hidden)`
**Kernel idea:** Fuse the SiLU activation and element-wise multiply into a single kernel. The two linears produce separate outputs that get multiplied — fuse the activation + multiply step.
**Frequency:** Once per block.

#### 3. Fused GELU + Dropout
**Source:** `src/models/utils/modules.py:153-158`
**Pattern:** `x = gelu(fc1(x)); x = dropout(x); x = fc2(x); x = dropout(x)`
**Kernel idea:** Fuse GELU + dropout into one kernel (save one read/write pass over the activation tensor).
**Frequency:** Once per block for non-SwiGLU models.

#### 4. Fused Loss Computation
**Source:** `app/vjepa_2_1/train.py:641`
**Pattern:** `torch.mean(torch.abs(zij - h_term) ** loss_exp) / loss_exp`
**Kernel idea:** Single-pass reduction: subtract, abs, pow, mean all in one kernel. Avoids materializing 3 intermediate tensors.
**Frequency:** Every training iteration, per mask group.

#### 5. Fused QKV Projection + Reshape
**Source:** `src/models/utils/modules.py` (Attention, RoPEAttention, ACRoPEAttention)
**Pattern:** `qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)` then unbind
**Kernel idea:** The reshape+permute after the linear creates a non-contiguous view. Fuse the permutation into a custom copy kernel that writes q, k, v directly into separate contiguous buffers.
**Frequency:** Every attention block.

#### 6. Fused Attention Output Projection + Residual
**Source:** `src/models/utils/modules.py` (after SDPA)
**Pattern:** `x = attn @ v; x = x.transpose(1,2).reshape(B,N,C); x = self.proj(x); x = self.proj_drop(x)`
**Kernel idea:** Fuse the transpose+reshape+linear into a single matmul with fused layout. Or at minimum fuse the dropout into the projection.

#### 7. Fused DropPath
**Source:** `src/models/utils/modules.py:118-140`
**Pattern:** Random sample-level drop applied to residual branch
**Kernel idea:** Fuse drop_path with the residual addition: `x = x + drop_path(y)` becomes a single kernel that reads x, y, generates random mask, and writes result.
**Frequency:** 2x per block.

### TIER 2 — Medium Impact

#### 8. Fused Sincos Position Embedding
**Source:** `src/models/utils/pos_embs.py`
**Pattern:** `np.sin/np.cos` grid construction on CPU then `.to(device)`
**Kernel idea:** Generate sincos embeddings directly on GPU with a Triton kernel. Avoid CPU-GPU transfer.

#### 9. Fused Patch Embedding Normalization
**Source:** `src/models/utils/patch_embed.py`
**Pattern:** Conv3D/Conv2D → flatten → transpose
**Kernel idea:** Fuse the flatten+transpose into a single contiguous output write.

#### 10. Fused Token Gathering with Positional Add
**Source:** `src/models/predictor.py:239-241`
**Pattern:** `argsort` + `gather` + positional embedding addition
**Kernel idea:** Combine the gather and pos_embed add into one kernel.

#### 11. Fused Multi-Head Attention Score + Mask
**Source:** `src/models/utils/modules.py` (non-SDPA fallback path)
**Pattern:** `(q @ k.T) * scale + softmax` with optional mask
**Kernel idea:** Fused attention with inline mask application (like FlashAttention but for the fallback path).

#### 12. Fused Cosine Distance
**Source:** V-JEPA loss uses L1/Lp distance. If cosine similarity variant exists:
**Pattern:** `F.normalize(a) @ F.normalize(b).T`
**Kernel idea:** Fuse normalize + matmul.

#### 13. Fused Gradient Clipping
**Source:** Training loop uses `torch.nn.utils.clip_grad_norm_`
**Pattern:** Compute global norm across all params, scale gradients
**Kernel idea:** Single-pass norm computation + in-place scaling.

#### 14. Fused AdamW Step
**Source:** Optimizer step
**Pattern:** Momentum update + weight decay + param update
**Kernel idea:** Fuse the 3 reads + 3 writes per parameter into 1 read + 1 write.

### TIER 3 — Speculative (try if time permits)

#### 15. Fused Mask Construction for Multi-Mask Training
**Source:** `src/masks/` mask generation
**Kernel idea:** Generate random masks directly on GPU instead of CPU.

#### 16. Fused Attention with RoPE Inline
**Source:** Combining RoPE rotation directly into the attention kernel
**Kernel idea:** Apply sin/cos rotation inside the attention tiling loop.

#### 17. Fused EMA Update
**Source:** Target encoder uses EMA of student weights
**Pattern:** `target_param.data = momentum * target_param.data + (1 - momentum) * student_param.data`
**Kernel idea:** Single kernel for all-parameter EMA update.

#### 18. Fused Video Tubelet Embedding
**Source:** `PatchEmbed3D` in `src/models/utils/patch_embed.py`
**Kernel idea:** Custom 3D conv kernel optimized for the specific tubelet sizes used.

#### 19. Fused Attention Softmax with Temperature
**Source:** Attention with learned or fixed temperature scaling
**Kernel idea:** Combine scale + softmax + dropout in one pass.

#### 20. Fused Concatenate + Linear
**Source:** Various places where `cat` is followed by a linear projection
**Kernel idea:** Write directly to the output of the linear without materializing the concatenated tensor.

## Rules

1. **One kernel per file.** No multi-kernel files.
2. **Always include baseline.** Copy the exact PyTorch code from source. Don't simplify it.
3. **Always include backward** if the operation is on the training path. Use `torch.autograd.Function`.
4. **Test with realistic shapes.** Use shapes from the actual V-JEPA 2 configs (ViT-L: embed_dim=1024, num_heads=16, depth=24; ViT-H: embed_dim=1280, num_heads=16, depth=32).
5. **Don't wait for results.** Write the kernel, test, bench, queue entry, move on. The queue runner handles execution.
6. **If a kernel idea is ambiguous, write two variants** as separate queue entries.
7. **Launch config matters.** For every Triton kernel, try at least `num_warps` in {2, 4, 8} and `BLOCK` sizes in {256, 512, 1024}. Submit the best guess first, then submit variants.
8. **fp16 is the default dtype.** V-JEPA 2 trains in mixed precision. Test fp16 and fp32.
9. **Guard everything.** The `can_use_kernel()` function must check device, contiguity, dtype, and shape constraints.
10. **No silent failures.** If the kernel can't run, fall back to baseline. Never crash.

## How To Add a New Kernel (Checklist)

```
[ ] Create src/models/utils/kernels/KERNELNAME.py with kernel_fn, baseline_fn, can_use_kernel, SHAPES
[ ] Create tests/queue/test_KERNELNAME.py with forward + backward parity tests
[ ] Create benchmarks/queue/bench_KERNELNAME.py with CUDA event timing
[ ] Append entry to queue/pending.jsonl
[ ] (Optional) Create a launch-config variant as a separate queue entry
```

## Target: 50+ kernels in the queue

You should be producing 5-10 kernels per hour. Each kernel is small (50-150 lines). The parity test is templated. The benchmark is templated. The bottleneck should be thinking about what to fuse, not writing boilerplate.

Go fast. The queue runner will tell you what survived.
