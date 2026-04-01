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
- separate block-level checks from short full-model checks,
- report neutral and rejected results, not just wins.

If a benchmark loads only one file from `HEAD` while other imports still come
from the working tree, say that explicitly. That is a useful check, but it is
not the same thing as a full repository replay.

Short-loop measurements here usually use:

- CUDA events,
- a warmup phase,
- `30-100` timed iterations,
- realistic `fp16` shapes on an RTX 3090.

The important caveat is that these short-loop measurements answer different
questions:

- a primitive benchmark tells you whether a local rewrite is even plausible,
- a block benchmark tells you whether the win survives real module wiring,
- a short full-model check tells you whether the default path actually moves.

Those are not interchangeable, and the docs below try to keep them separate.

## Test Strategy

The fast validation stack is:

- primitive parity tests,
- block parity tests,
- small model/predictor parity tests,
- targeted correctness tests for eval determinism and mixed precision,
- optional manual-kernel checks when a Triton experiment is being evaluated.

Recent retained coverage includes:

- [`tests/models/test_attention_correctness.py`](/workspace/vjepa2/tests/models/test_attention_correctness.py)
- [`tests/test_predictor_parity.py`](/workspace/vjepa2/tests/test_predictor_parity.py)
- [`tests/test_ac_predictor_parity.py`](/workspace/vjepa2/tests/test_ac_predictor_parity.py)
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
- separated-position caching,
- `einsum` removal in favor of broadcasted multiply,
- compatibility duplication via `torch.cat`,
- fused `q/k` rotation reuse.

These wins are real, but the honest story is mixed:

- they help RoPE blocks,
- some are close to neutral at short full-model scale,
- the action-conditioned path benefits more than the plain encoder path.

That is still useful, but it is not a license to overclaim.

### Faster Multi-Mask `apply_masks`

The retained `apply_masks` win is not the rejected Triton kernel. It is a plain
PyTorch vectorization that only triggers on the honest common case:

- multiple masks,
- every mask is 2D,
- every mask has the same shape.

Instead of launching one `gather` per mask group, the masks are stacked and
handled by one batched `gather`.

Fresh primitive result on `[32, 1024, 384]`, `4` mask groups of shape
`[32, 256]`, `fp16`, CUDA:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `apply_masks_multi_2d` | 0.2104 ms | 0.1096 ms | 1.92x |

Why it stayed:

- parity is easy to verify,
- the speedup is large enough to matter,
- it improves the real helper without introducing custom-kernel risk.

### Same-Shape 1D `apply_masks`

The same batched-gather idea also helps when every mask is 1D and the same
length. That is a separate fast path from the 2D case, and it still shows up in
helper-style call sites.

Fresh primitive result on `[32, 1024, 384]`, `4` mask groups of length `256`,
`fp16`, CUDA:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `apply_masks_multi_1d` | 0.2736 ms | 0.1323 ms | 2.07x |

Why it stayed:

- it uses the same exact batching idea as the 2D path,
- parity is straightforward,
- it is an extra win, not a replacement for the 2D result.

### Rejected: Predictor RoPE Position Precompute

One useful lesson from this pass is that not every worthwhile win needs a custom
kernel. In the masked RoPE predictor path, the sorted token ids are the same for
every predictor block in a forward pass, but the old code still decomposed them
into frame/height/width positions on every block.

The first attempt was simple:

- compute the separated RoPE positions once in the predictor,
- pass them into each RoPE block,
- keep the attention math identical.

That looked promising initially, but after fixing the surrounding correctness
issues and benchmarking against the corrected dynamic path, the precompute
itself lost:

| benchmark | dynamic corrected path | precomputed variant | speedup |
| --- | ---: | ---: | ---: |
| `predictor_rope_internal` | 13.9165 ms | 14.2865 ms | 0.973x |

So the right decision was:

- reject the precompute optimization,
- keep the correctness fixes around the same path,
- stop describing this area as a retained throughput win.

What was retained instead:

- non-square masked RoPE predictor inputs now use the real `H/W` grid instead of
  silently falling back to square behavior,
- `has_cls=True` no longer crashes under RoPE,
- both of those behaviors are now covered by targeted tests.

### SDPA Correctness Fixes

The attention modules now avoid passing dropout into SDPA during `eval()`, and
the live RoPE path no longer breaks half precision just because cached
positions were created in `float32`.

These are correctness fixes first, performance changes second.

Why they matter:

- eval output is now deterministic when it should be,
- half-precision RoPE attention works on the main CUDA path,
- optimization claims are worthless if the path is wrong.

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
- any narrower forward-only Triton path must still earn its place with an
  independent reference check,
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

### Rejected: Triton `apply_masks` Gather Kernel

A fused Triton gather kernel for [`apply_masks`](/workspace/vjepa2/src/masks/utils.py)
was prototyped with exact backward semantics via scatter-add.

What looked good:

- direct primitive benchmark improved,
- backward parity matched exactly,
- the kernel was straightforward and safe enough to test.

Measured primitive result on `[8, 1024, 384]`, `4` mask groups, `fp16`, CUDA:

| kernel | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `apply_masks` | 0.2958 ms | 0.2611 ms | 1.13x |

Why it was still rejected:

- masked image encoder check: `52.0714 ms -> 52.1849 ms`
- masked video encoder check: `40.1243 ms -> 40.2205 ms`

So the primitive win was real, but it did not survive a real caller. That means
it does not belong in the live path.

## Fused `q/k` RoPE Pair Kernel

The next retained win came from the repeated `rotate_query_key_pair` path.
This helper rotates query and key tensors with the same positions in RoPE
attention.

The old implementation was correct but indirect:

- concatenate `q` and `k`,
- rotate the packed tensor,
- split the result back out.

The retained Triton kernel rotates both tensors together in one pass:

- one position load,
- one sin/cos evaluation,
- two outputs written together.

Primitive benchmark on `[8, 16, 4096, 24]`, `fp16`, CUDA:

| kernel | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `rotate_query_key_pair` | 1.3617 ms | 0.2182 ms | 6.24x |

That is the kind of local win that is worth keeping.

The caller-level check is the real filter. Under `no_grad`, the kernel is used
in inference, and the measured results were:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `ac_rope_attention_forward` | 2.0785 ms | 1.9283 ms | 1.08x |
| `rope_attention_forward` | 1.1694 ms | 1.2025 ms | 0.97x |

So the rule stays the same:

- keep the kernel where the caller improves,
- reject it where the caller does not.

In this case, the fused pair kernel survives because it helps the action-
conditioned RoPE path, while the plain RoPE caller path on this benchmark shape
does not earn a claim by itself.

## Two Useful Rejections

These are worth teaching explicitly because both ideas sounded reasonable, both
passed parity, and both still failed the real keep/reject filter.

### Contiguous Patch-Embed Output

The patch-embed modules return `flatten(...).transpose(1, 2)`, which is a
non-contiguous token layout. The obvious experiment is to pay that copy once and
return a contiguous `[B, N, C]` tensor into the encoder stack.

That hypothesis only half-worked.

Short encoder checks on CUDA, `fp16`, `depth=4`:

| benchmark | baseline | contiguous variant | speedup |
| --- | ---: | ---: | ---: |
| image unmasked | 3.8271 ms | 3.7288 ms | 1.03x |
| image masked | 3.7480 ms | 3.7693 ms | 0.99x |
| video unmasked | 3.8688 ms | 3.5719 ms | 1.08x |
| video masked | 3.6584 ms | 3.7410 ms | 0.98x |

So the lesson is not "non-contiguous is always bad." The lesson is:

- the copy can help later blocks,
- the copy can also hurt the masked training path,
- if the main path loses, the optimization is rejected.

### Direct Predictor Mask-Token Expansion

Another plausible idea was to skip the full `[B, num_patches, D]` expansion for
mask tokens in the predictor and expand directly to the masked target shape.

That variant was correct after switching the positional add to an out-of-place
form, but it still lost:

| benchmark | baseline | direct-expand variant | speedup |
| --- | ---: | ---: | ---: |
| predictor mask-token path | 6.6483 ms | 6.7706 ms | 0.98x |

So this one joins the rejected set as well.

## Porting A Win Across Trees

One of the most productive later passes was not a brand-new kernel. It was
finding code drift between `src` and `app/vjepa_2_1`.

The app-side predictor still had the older pattern:

- explicit `repeat(...)`,
- Python row-wise permutation with `torch.stack([...])`,
- extra copies where `torch.gather(...)` could apply the same permutation.

That is exactly the kind of cleanup that is easy to underestimate because it
does not look exotic. But it is often the highest-confidence work left after the
first obvious wins land.

The ported cleanup did four things:

- replaced safe `repeat(...)` calls with `expand(...)`,
- replaced the repeated context copy with `unsqueeze + expand + reshape`,
- replaced row-wise Python permutation with `torch.gather(...)`,
- broadcast modality embeddings instead of materializing repeated copies.

That kept the math identical and moved the app-side predictor forward path from:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `app_predictor_forward` | 23.3783 ms | 21.8536 ms | 1.07x |

The important lesson is that "new optimization work" does not always mean "new
idea." Sometimes the best next win is applying a proven idea consistently across
the parts of the codebase that still lag behind.

## App-Side RoPE Drift Cleanup

The same pattern showed up again in the app RoPE stack.

The older app tree was still:

- recomputing separated depth/height/width positions in the no-mask path,
- rotating `q` and `k` independently for each RoPE axis,
- missing the pair-oriented cleanup that had already proved itself elsewhere.

The retained fix stayed conservative:

- cache separated positions,
- add a pair-oriented `rotate_query_key_pair(...)`,
- route app `RoPEAttention` and `Block` through that pair helper.

This was not a new autograd trick. It was just removing duplicated tensor work.

Measured against `HEAD` on CUDA:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `rotate_query_key_pair` | 0.9752 ms | 0.5928 ms | 1.65x |
| `rope_attention_forward` | 4.5426 ms | 2.4995 ms | 1.82x |
| `rope_block_forward` | 5.0050 ms | 2.8540 ms | 1.75x |

And at the app predictor level, the cumulative effect now shows up too:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `app_predictor_forward` | 16.3713 ms | 14.7981 ms | 1.11x |

This is exactly the kind of optimization work that compounds well:

- it is local,
- parity is easy to lock down,
- the caller-level win is large enough to matter,
- and it improves a real model path rather than a synthetic primitive only.

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
3. Add manual kernels only for small, repeated primitives around attention after
   a plain PyTorch rewrite stops winning.
4. Never trust a forward-only win on a training path without gradient checks.
5. Keep a permanent record of rejected experiments so you do not rediscover the
   same dead ends.

## Current Best Next Targets

The best remaining candidates are:

1. Action-conditioned path cleanup where `cat` and small tensor glue still show
   up in profiles.
2. Masked-path tensor glue beyond the current `apply_masks` and predictor RoPE
   wins, but only if it survives caller-level checks.
3. Larger model-level forward/backward measurements on real training shapes to
   confirm which block-level wins matter.

## Commands

Representative retained validation:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/models/test_attention_correctness.py \
  tests/test_predictor_parity.py \
  tests/test_kernel_parity.py \
  tests/models/test_ac_rope_attention.py \
  tests/test_model_block_parity.py
```

## Final Rule

The discipline that matters is:

- baseline first,
- parity first,
- keep only what survives both.

That rule is stricter than "write lots of kernels", but it is how you actually
end up with a faster codebase instead of a fragile one.
