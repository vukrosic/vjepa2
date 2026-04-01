# How To Optimize V-JEPA 2 Without Lying To Yourself

This document is the cleaned-up version of the optimization work in this repo.
The running diary stays in [`OPTIMIZATION_NOTES.md`](/workspace/vjepa2/OPTIMIZATION_NOTES.md).
This file is the tutorial: what to optimize, how to test it, how to benchmark
it fairly, and how to decide whether a change should survive.

## Goal

The goal is simple:

- make V-JEPA 2 faster,
- do not change model behavior,
- keep only changes that beat a real baseline.

That last point matters more than people admit. Plenty of optimization work
produces a prettier microbenchmark and a worse real model.

## What Counts As A Real Optimization

A change only counts if it clears all of these:

1. It preserves functionality.
2. It has a parity test or an exact reference check.
3. It is measured against the checked-in baseline or a faithful local baseline.
4. It wins on the path that actually matters.
5. It does not silently break training, mixed precision, or eval behavior.

This repo now uses that standard consistently. A fast but wrong Triton kernel is
not a win. A model-block win that vanishes in the full model is not a headline
win. A fallback-path cleanup that never touches the default path is not a main
result.

## The Workflow

The repeatable loop is:

1. Profile or inspect the hot path.
2. Form one narrow hypothesis.
3. Patch one thing.
4. Add the smallest useful parity check.
5. Run a short benchmark against baseline.
6. Keep or reject immediately.

That loop is intentionally short. Long benchmark suites are valuable later, but
they are bad for exploration. Fast iteration is how you discover which ideas are
worth deeper validation.

## Benchmarking Rules

Fair benchmarking in this repo means:

- compare against `HEAD` when possible,
- otherwise compare against a faithful reconstructed baseline,
- use the same shapes, dtype, device, and mode,
- separate microbenchmarks from block-level checks,
- report neutral and rejected results, not just wins.

Short-loop measurements here usually use:

- CUDA events,
- a warmup phase,
- `30-100` timed iterations,
- realistic `fp16` shapes on an RTX 3090.

## Test Strategy

The fast validation stack is:

- primitive parity tests,
- block parity tests,
- small model/predictor parity tests,
- targeted correctness tests for eval determinism and mixed precision,
- Triton-specific forward checks where manual kernels are retained.

Recent retained coverage includes:

- [`tests/models/test_attention_correctness.py`](/workspace/vjepa2/tests/models/test_attention_correctness.py)
- [`tests/models/test_triton_rope_kernel.py`](/workspace/vjepa2/tests/models/test_triton_rope_kernel.py)
- [`tests/test_predictor_parity.py`](/workspace/vjepa2/tests/test_predictor_parity.py)
- [`tests/test_kernel_parity.py`](/workspace/vjepa2/tests/test_kernel_parity.py)
- [`tests/test_model_block_parity.py`](/workspace/vjepa2/tests/test_model_block_parity.py)

## Where Time Was Actually Going

The useful hotspots were not mysterious:

- repeated Python-side mask and tensor assembly,
- repeated RoPE frequency/position work,
- extra tensor replication and concatenation,
- action-conditioned attention glue around otherwise efficient kernels.

The heavy attention core is already mostly handled by PyTorch SDPA. That means
the best opportunities were usually in the code around attention, not in trying
to out-GEMM cuBLAS.

## Retained Optimizations

These are the changes worth keeping.

### Cached Action Causal Mask

Instead of rebuilding the action-block causal mask every call, the result is
cached and cloned on demand.

Result from the retained benchmark:

| kernel | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| action causal mask build | 24.80 ms | 0.78 ms | 31.79x |

Why it stayed:

- exact behavior is easy to verify,
- the win is large,
- there is no training or numerical risk.

### Faster Batch Repeat-Interleave

`repeat_interleave_batch` was rewritten around reshape/expand instead of nested
concats.

Result:

| kernel | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `repeat_interleave_batch` | 189.03 ms | 99.48 ms | 1.90x |

Why it stayed:

- exact parity is straightforward,
- the speedup is consistent,
- this pattern appears in real model code.

### RoPE Math Cleanup

Several RoPE-side changes survived:

- inverse-frequency caching,
- cached positions,
- `einsum` removal in favor of broadcasted multiply,
- compatibility duplication via `torch.cat`,
- fused `q/k` rotation reuse.

These wins are real, but the honest story is mixed:

- they help RoPE blocks,
- they are often neutral at full-model scale,
- the action-conditioned path benefits more than the plain encoder path.

That is still useful, but it is not a license to overclaim.

### SDPA Correctness Fixes

The attention modules now avoid passing dropout into SDPA during `eval()`, and
the live RoPE path no longer breaks half precision just because cached
positions were created in `float32`.

These are correctness fixes first, performance changes second.

Why they matter:

- eval output is now deterministic when it should be,
- half-precision RoPE attention works on the main CUDA path,
- optimization claims are worthless if the path is wrong.

### Manual Kernel Kept: Forward-Only Triton RoPE Rotate

The repo retains a manual Triton kernel for the single-axis unmasked RoPE rotate
primitive, but only for safe forward/no-grad execution.

Measured on `[8, 16, 1024, 64]`, `fp16`, CUDA:

| kernel | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `rotate_queries_or_keys` forward | 0.5608 ms | 0.1989 ms | 2.82x |

Why it stayed:

- forward parity is checked against a pure PyTorch reference,
- the live training path falls back to PyTorch,
- gradient parity is exact on the live helper because autograd no longer routes
  through the Triton kernel.

This is the right way to keep a manual kernel that is useful for inference-style
execution without pretending it is training-safe when it is not.

## Rejected Experiments

This section matters as much as the retained wins.

### Rejected: Triton RoPE Autograd Path

The first version of the Triton path tried to support backward by applying the
same rotate with `-pos`.

Why it was rejected:

- the repo uses a compatibility-preserving RoPE layout,
- the backward was not mathematically valid for that layout,
- gradient parity failed badly.

Decision:

- no live Triton autograd path,
- forward-only Triton is allowed,
- training stays on the PyTorch reference path.

### Rejected: Fused Multi-Axis Q/K Triton Kernel

A fused Triton kernel for depth/height/width RoPE rotation looked like the next
obvious step.

Why it was rejected:

- forward parity failed on realistic shapes,
- the kernel looked fast but was not correct enough,
- the failure happened before it earned the right to any larger benchmark.

### Rejected: Batched-Position Triton Extension

Extending the single-axis Triton kernel to batched masked positions produced an
attractive microbenchmark.

Why it was rejected:

- parity failed on masked-path shapes,
- the apparent speedup was not trustworthy,
- masked training paths are exactly where you cannot tolerate silent numeric
  drift.

### Rejected: Predictor Full-Merge Reorder

The predictor had a plausible optimization candidate: replace the sort/reverse
sort path with a more specialized merge.

Why it was rejected:

- the full merge did not clearly beat the current path,
- the narrower “target positions only” cleanup was safer,
- end-to-end predictor gains were tiny.

## A Good Optimization vs A Bad One

Good optimization:

- cached action masks,
- exact semantics,
- large win,
- obvious keep.

Bad optimization:

- fused Triton multi-axis RoPE kernel,
- impressive idea,
- fast-looking microbench,
- failed parity,
- immediate reject.

That contrast is the actual lesson. Optimization is not about cleverness. It is
about surviving the keep/reject filter.

## Practical Advice

If you want to optimize a model like this:

1. Start with Python loops, repeated tensor construction, and needless copies.
2. Treat SDPA as the baseline winner until profiling proves otherwise.
3. Add manual kernels only for small, repeated primitives around attention.
4. Never trust a forward-only win on a training path without gradient checks.
5. Keep a permanent record of rejected experiments so you do not rediscover the
   same dead ends.

## Current Best Next Targets

The best remaining candidates are:

1. A fused `apply_masks` manual gather kernel on the masked encoder path.
2. Action-conditioned path cleanup where `cat` and small tensor glue still show
   up in profiles.
3. Larger model-level forward/backward measurements on real training shapes to
   confirm which block-level wins matter.

## Commands

Representative retained validation:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/models/test_triton_rope_kernel.py \
  tests/models/test_attention_correctness.py \
  tests/test_predictor_parity.py \
  tests/test_kernel_parity.py \
  tests/models/test_ac_rope_attention.py \
  tests/test_model_block_parity.py
```

Representative manual-kernel measurement:

- pure PyTorch RoPE rotate baseline vs Triton forward kernel on CUDA `fp16`
- live gradient parity check through [`rotate_queries_or_keys`](/workspace/vjepa2/src/models/utils/modules.py)

## Final Rule

The discipline that matters is:

- baseline first,
- parity first,
- keep only what survives both.

That rule is stricter than "write lots of kernels", but it is how you actually
end up with a faster codebase instead of a fragile one.
