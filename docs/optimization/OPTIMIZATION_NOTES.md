# V-JEPA 2 Optimization Notes

This file is a running engineering note for the optimization work in this repo.

It is intentionally written like a tutorial blog post, not a terse changelog.
The goal is to make each speedup reproducible, explain why it is safe, and show
how to measure it against a baseline before keeping it.

One important framing rule for the whole document:

- primitive wins are not automatically block wins,
- block wins are not automatically default-path model wins,
- short forward-only checks are useful for exploration, but they are not enough
  to claim a broad training-speed result on their own.

## Scope

Constraints for this pass:

- No intended functionality changes.
- Every optimization needs a baseline.
- Every retained optimization needs parity coverage.
- Every speculative optimization should be benchmarked and dropped if it loses.

Hardware used for the measurements below:

- GPU: NVIDIA GeForce RTX 3090
- PyTorch: `2.11.0+cu126`
- CUDA capability: `sm_86`

## Optimization Workflow

The pattern used in this repo is:

1. Find a hot path or an obviously inefficient tensor/Python loop.
2. Write down the baseline implementation.
3. Replace it with a vectorized or fused version.
4. Add a parity test.
5. Add a benchmark that measures baseline vs optimized directly.
6. Keep the change only if it wins and still passes tests.

That sounds obvious, but it matters. A lot of "optimization" work just moves code
around and makes it harder to understand without improving throughput.

### How Baselines Are Measured

The comparison script in [`benchmarks/baseline_compare.py`](/workspace/vjepa2/benchmarks/baseline_compare.py)
loads the version of a file from `HEAD` with `git show`, writes it to a temporary
module, and runs the baseline and working-tree functions on the same random
inputs.

This is the important part:

- It compares against the checked-in baseline, not a hand-copied approximation.
- It uses the same tensor shapes and the same device for both paths.
- It runs a warmup phase before timing.
- It reports both a human-readable markdown table and JSON.
- It includes at least one small model block, not just primitive kernels.
- The current comparator covers `build_action_block_causal_attention_mask`,
  `rotate_queries_or_keys`, `repeat_interleave_batch`, `apply_masks`, and
  `attention_block_forward`.
- The predictor comparison is intentionally narrower: it loads
  [`src/models/predictor.py`](/workspace/vjepa2/src/models/predictor.py) from
  `HEAD`, but its shared imports still resolve to the working tree. Treat
  `predictor_rope_forward` as a targeted file-level comparison, not a full repo
  replay.

That last point matters because some changes only look good at the primitive
level but disappear once the whole attention block is measured.

## Step 1: SDPA Attention Policy

### Why this was examined

The attention code already used `scaled_dot_product_attention`, but backend
selection was implicit. On Ampere, backend choice can change the result a lot,
especially for masked vs unmasked attention.

There was also a correctness problem: if attention dropout was configured,
`dropout_p` could still be passed into SDPA during `eval()`. For SDPA, dropout is
controlled by the explicit argument, not by module mode alone.

### What changed

In both model trees:

- [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py)
- [`app/vjepa_2_1/models/utils/modules.py`](/workspace/vjepa2/app/vjepa_2_1/models/utils/modules.py)

I added a small SDPA wrapper that:

- chooses backend priority explicitly,
- prefers fused kernels when they are legal,
- skips flash for masked attention,
- falls back to math safely,
- forces `dropout_p=0.0` during eval.

### How it was benchmarked

Benchmark CLI:

```bash
cd /workspace/vjepa2
python benchmarks/attention_sdpa.py --iters 30 --warmup-iters 10 --repeats 3 --check-parity
```

The benchmark compares:

- `default_auto`: baseline behavior, let PyTorch choose
- `optimized`: explicit backend priority
- `math_only`: sanity baseline

### Current result

Shape: `8 x 16 x 1024 x 64`, dtype `fp16`

| case | baseline `default_auto` | optimized | speedup | note |
| --- | ---: | ---: | ---: | --- |
| masked | 0.8488 ms | 0.8410 ms | 1.009x | small win |
| unmasked | 1.1101 ms | 1.1110 ms | 0.999x | basically neutral |

Interpretation:

- This is not a strong enough throughput win to call it a headline optimization.
- It is still worth keeping because it fixes the eval-mode dropout bug and makes
  backend choice explicit and testable.

### Supporting tests

- [`tests/models/test_sdpa_attention.py`](/workspace/vjepa2/tests/models/test_sdpa_attention.py)
- [`tests/models/test_attention_kernels.py`](/workspace/vjepa2/tests/models/test_attention_kernels.py)
- [`tests/support/test_kernel_benchmark.py`](/workspace/vjepa2/tests/support/test_kernel_benchmark.py)

## Step 2: Action Causal Mask Construction

### Why this was examined

The original mask builder in [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py)
used nested Python loops to fill block regions:

```python
for t1 in range(T):
    for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
        mask[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = mask_block
```

That is correct, but it is Python-heavy and writes many slices repeatedly.

### Optimization idea

The mask is really just a lower-triangular frame-level matrix, expanded into
token blocks. So instead of filling blocks manually, build the frame mask once
and expand it with `repeat_interleave`.

### New implementation shape

```python
frame_mask = torch.ones((T, T), dtype=torch.bool).tril()
return frame_mask.repeat_interleave(N_T, dim=0).repeat_interleave(N_T, dim=1)
```

### Benchmark

```bash
cd /workspace/vjepa2
python benchmarks/kernel_speedups.py --warmup-iters 10 --iters 50
```

### Result

| kernel | baseline | optimized | speedup | improvement |
| --- | ---: | ---: | ---: | ---: |
| action causal mask build | 43.61 ms | 2.17 ms | 20.07x | 95.02% |

Note:

- This benchmark is CPU-side and somewhat noisy.
- Even with noise, the win is large enough that the conclusion is stable.

### Why this is safe

- The parity test compares the optimized version to the original loop behavior.
- The mask semantics are unchanged: still a full lower-triangular block mask.

## Step 3: `repeat_interleave_batch`

### Why this was examined

The original implementation in [`src/utils/tensors.py`](/workspace/vjepa2/src/utils/tensors.py)
was pure Python concatenation:

```python
N = len(x) // B
x = torch.cat([torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)], dim=0)
return x
```

That creates a lot of Python overhead and many temporary tensors.

### Optimization idea

Reshape into `(N, B, ...)`, add a repeat axis, expand it, then reshape back.
This keeps the same order but avoids the nested Python `cat`.

### New implementation shape

```python
N = len(x) // B
return x.reshape(N, B, *x.shape[1:]).unsqueeze(1).expand(N, repeat, B, *x.shape[1:]).reshape(
    N * B * repeat, *x.shape[1:]
)
```

### Result

| kernel | baseline | optimized | speedup | improvement |
| --- | ---: | ---: | ---: | ---: |
| `repeat_interleave_batch` | 186.69 ms | 96.76 ms | 1.93x | 48.17% |

### Why this is safe

- A parity test compares output order exactly against the original implementation.
- An earlier draft got the repeat order wrong, and the test caught it. That is a
  good example of why these tests matter.

## Step 4: Experiments That Were Rejected

Not every optimization should be kept.

### RoPE rotation rewrite

I tried rewriting the RoPE rotation helpers to work directly in pair-space and
avoid some expansion overhead. The rewrite passed basic reasoning checks, but it
did not beat the baseline on the 3090.

Observed result from the exploratory benchmark:

- `rope_rotate_src_cuda_fp16`: slightly slower than baseline
- `rope_rotate_app_cuda_fp16`: slightly slower than baseline

Conclusion:

- reverted from the retained optimization set,
- benchmark code keeps an exploratory mode if I want to revisit it later.

This is an important point: many "clever" tensor rewrites do not help once the
actual kernel launch and memory behavior are measured.

### `apply_masks` broadcast rewrite

This one initially looked risky because the repo mixes 1D and 2D mask shapes.
After tightening the shape handling and adding parity tests, the optimization was
kept.

Current shape-safe approach:

- keep 1D masks working,
- keep 2D masks working,
- avoid materializing a repeated index tensor when `expand` is enough.

### `ac_predictor` device-local attention-mask cache

I also tried caching the already-built causal attention mask on the target
device inside [`src/models/ac_predictor.py`](/workspace/vjepa2/src/models/ac_predictor.py).

The idea was simple:

- avoid slicing the CPU mask every forward,
- avoid `.to(device)` every forward,
- reuse the same `(seq_len, device)` mask tensor.

This was parity-correct, but it did not survive the short GPU benchmark.

Observed result from the exploratory check:

| benchmark | baseline | cached-mask variant | speedup |
| --- | ---: | ---: | ---: |
| `ac_predictor forward` | 13.9570 ms | 14.0948 ms | 0.990x |

Conclusion:

- reverted,
- documented here so I do not re-run the same weak idea later.

### `ac_predictor` token preallocation

Another plausible idea was to replace:

```python
torch.cat([a, s, x], dim=2).flatten(1, 2)
```

and the extrinsics variant with a preallocated output tensor plus slice writes.

That sounds good on paper, but the direct GPU microbenchmark said otherwise.

Observed result from the isolated tensor benchmark:

| case | baseline | preallocated variant | speedup |
| --- | ---: | ---: | ---: |
| no extrinsics | 0.1250 ms | 0.1367 ms | 0.915x |
| with extrinsics | 0.1270 ms | 0.1669 ms | 0.761x |

Conclusion:

- rejected before patching the model,
- `torch.cat(...).flatten(...)` is already the better kernel path here.

### `ACRoPEAttention` merge preallocation

I also tested replacing the action/frame merge in `ACRoPEAttention`:

```python
torch.cat([ta, tx], dim=3).flatten(2, 3)
```

with a preallocated output tensor and slice writes.

Again, the direct benchmark said no.

Observed result from the isolated tensor benchmark:

| shape | baseline | preallocated variant | speedup |
| --- | ---: | ---: | ---: |
| `B=2, H=8, T=4, HW=64, A=2, D=32` | 0.0770 ms | 0.0955 ms | 0.806x |
| `B=4, H=16, T=8, HW=196, A=2, D=64` | 0.0761 ms | 0.0966 ms | 0.788x |

Conclusion:

- rejected before touching the main forward path,
- the baseline `cat(...).flatten(...)` sequence is already stronger here.

### Predictor inverse-permutation via scatter

One more plausible predictor idea was to replace:

```python
reverse_argsort = torch.argsort(argsort, dim=1)
```

with a direct inverse-permutation build using `scatter`.

This is algorithmically attractive, but the short GPU benchmark said the current
second `argsort` is still faster at the tested sizes.

Observed result from the isolated tensor benchmark:

| shape | baseline | scatter variant | speedup |
| --- | ---: | ---: | ---: |
| `B=8, N=512` | 0.0953 ms | 0.1078 ms | 0.884x |
| `B=32, N=512` | 0.0948 ms | 0.1098 ms | 0.864x |
| `B=32, N=1024` | 0.0984 ms | 0.1096 ms | 0.898x |
| `B=64, N=1024` | 0.0964 ms | 0.1104 ms | 0.873x |

Conclusion:

- rejected before patching the predictor,
- the baseline path stays.

### RoPE pair rotation with slice assignment

I also tried replacing the pair rotation:

```python
y = x.unflatten(-1, (-1, 2))
y1, y2 = y.unbind(dim=-1)
y = torch.stack((-y2, y1), dim=-1).flatten(-2)
```

with direct slice assignment into an `empty_like(x)` buffer.

That version was parity-clean, but the win was too small to justify the extra
implementation complexity once the better `cat` improvement was already in
place.

Observed result from the isolated tensor benchmark:

| benchmark | baseline | slice-write variant | speedup |
| --- | ---: | ---: | ---: |
| `rotate_queries_or_keys` helper | 0.6358 ms | 0.6294 ms | 1.010x |

Conclusion:

- rejected,
- not enough return for the added code complexity.

### Quick profiler sanity checks

At this point I stopped guessing for a moment and ran short one-forward GPU
profiles on the real model paths.

What the masked encoder profile showed:

- most CUDA time was already in GEMMs and fused attention,
- `gather` and mask handling were small,
- that made more encoder-side tensor rewrites a low-probability target.

What the direct `ACRoPEAttention` profile showed:

- `aten::cat`,
- `aten::mul`,
- `aten::sin`,
- `aten::cos`

were still meaningful self-CUDA costs inside the RoPE-heavy block.

That was the signal to stop chasing more Python-level reshapes and try a fused
kernel instead.

## Step 5: Batched Action-Token QKV In `ACRoPEAttention`

### Why this was examined

`ACRoPEAttention` had a very visible small-kernel pattern: when `action_tokens >
0`, it looped over each action token and ran a separate QKV projection plus RoPE
rotation for each one.

That pattern is expensive because:

- it launches more small kernels than necessary,
- it repeats the same setup work per token,
- it adds Python loop overhead to a path that can be batched.

### Original shape

The old code effectively did:

```python
for i in range(action_tokens):
    a = x[:, :, i : i + 1, :].flatten(1, 2)
    qkv = self.qkv(a) ...
    ...
```

### Optimization idea

Instead of projecting each action token independently, flatten all action tokens
for all timesteps in the existing order, run one QKV projection, rotate the
depth component once on the flattened action sequence, then reshape back.

This keeps the same token order:

- time-major,
- then action-token order within each timestep.

### How it was validated

I added a direct parity test:

- [`tests/models/test_ac_rope_attention.py`](/workspace/vjepa2/tests/models/test_ac_rope_attention.py)

The test compares the current module output against a local reference function
that implements the old loop behavior step by step.

Quick validation command:

```bash
cd /workspace/vjepa2
pytest -q tests/models/test_ac_rope_attention.py tests/test_kernel_parity.py
```

### Working-tree vs `HEAD` benchmark

This is now measured in the main baseline harness instead of a one-off check:

```bash
cd /workspace/vjepa2
python benchmarks/baseline_compare.py
```

Current result:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `ac_rope_attention_forward` | 10.0614 ms | 6.2051 ms | 1.62x |

That is a real model-adjacent win, not just a helper-level microbenchmark.

## Step 6: Working-Tree vs `HEAD` Block Benchmarks

Primitive benchmarks are useful, but they are not enough.

To make sure the helper wins survive contact with the real forward path, the
current pass also benchmarks the working tree directly against `HEAD` for:

- primitive kernels,
- the plain `Attention` block,
- `RoPEAttention`,
- `ACRoPEAttention`.

Current command:

```bash
cd /workspace/vjepa2
python benchmarks/baseline_compare.py
```

Current result:

| name | baseline ms | optimized ms | speedup |
| --- | ---: | ---: | ---: |
| `build_action_block_causal_attention_mask` | 2.2775 | 0.2343 | 9.72x |
| `attention_block_forward` | 0.4482 | 0.4256 | 1.05x |
| `rotate_queries_or_keys` | 0.6987 | 0.5337 | 1.31x |
| `repeat_interleave_batch` | 4.2359 | 0.0724 | 58.47x |
| `apply_masks` | 0.2784 | 0.1418 | 1.96x |
| `ac_rope_attention_forward` | 10.0614 | 6.2051 | 1.62x |
| `rope_attention_forward` | 5.3331 | 4.3442 | 1.23x |

The important takeaway is not the exact decimal. It is that the kept helper
optimizations do show up in real RoPE-heavy attention blocks, especially in the
action-conditioned path.

## Step 7: Predictor Broadcast And Gather Cleanup

### Why this was examined

After the RoPE-heavy attention path, the next easy target was the predictor.
The `src` predictor still had a set of cheap-but-frequent tensor operations:

- positional embeddings copied with `repeat`,
- mask tokens copied with `repeat`,
- context tokens duplicated with `repeat`,
- token reorder implemented by Python loops plus `torch.stack`.

## Step 8: Multi-Mask `apply_masks` Fast Path

### Why this was examined

After the earlier rejected Triton `apply_masks` experiment, the primitive itself
was still clearly worth revisiting in plain PyTorch.

The live helper in [`src/masks/utils.py`](/workspace/vjepa2/src/masks/utils.py)
still did a Python loop over mask groups and launched one `gather` per mask:

```python
all_x = []
for m in masks:
    mask_keep = m.unsqueeze(-1).expand(*m.shape, x.size(-1))
    all_x += [torch.gather(x, dim=1, index=mask_keep)]
return torch.cat(all_x, dim=0)
```

That is fine for one mask group. It is wasteful for the common case where:

- every mask is 2D,
- every mask has the same shape,
- the only difference is which indices each group selects.

### Optimization idea

If every mask has shape `[B, K]`, stack them into `[M, B, K]`, broadcast `x`
once to `[M, B, N, D]`, and issue a single batched `gather`.

The retained implementation keeps all old behavior:

- empty mask lists still return a valid empty result,
- the single-mask case stays on the simple path,
- 1D masks keep the old fallback behavior,
- `concat=False` still returns a list.

Only the same-shape multi-2D case takes the new stacked fast path.

### Benchmark

Primitive benchmark command:

```bash
cd /workspace/vjepa2
python benchmarks/kernel_speedups.py
```

Fresh result on `[32, 1024, 384]`, `4` mask groups of shape `[32, 256]`, `fp16`, CUDA:

| benchmark | baseline | optimized | speedup | improvement |
| --- | ---: | ---: | ---: | ---: |
| `apply_masks_multi_2d` | 0.2104 ms | 0.1096 ms | 1.92x | 47.89% |

The same stacked-gather trick also wins for same-shape 1D masks.

Fresh result on `[32, 1024, 384]`, `4` mask groups of length `256`, `fp16`, CUDA:

| benchmark | baseline | optimized | speedup | improvement |
| --- | ---: | ---: | ---: | ---: |
| `apply_masks_multi_1d` | 0.2736 ms | 0.1323 ms | 2.07x | 51.64% |

### Why it was kept

- It is a real win on the surviving live shape, not only on a synthetic helper.
- It avoids the complexity of the rejected Triton path.
- Parity is straightforward and now explicitly covered for:
  - CPU 2D masks,
  - CUDA 2D masks,
  - `concat=False`,
  - the old fallback behavior.

Supporting coverage lives in:

- [`tests/test_kernel_parity.py`](/workspace/vjepa2/tests/test_kernel_parity.py)
- [`benchmarks/kernel_speedups.py`](/workspace/vjepa2/benchmarks/kernel_speedups.py)

## Step 9: Rejected Predictor RoPE Position Precompute, Retained Correctness Fixes

### Why this was examined

In the RoPE predictor path, the sorted `masks` tensor is identical for every
predictor block in the forward pass. But each `RoPEAttention` block was still
recomputing:

- frame ids,
- height ids,
- width ids

from the same sorted token ids every time.

That work is cheap once, but not cheap when repeated across every block. A short
microcheck on the live helper put one `separate_positions(mask.unsqueeze(1))`
call at about `0.2622 ms` on the 3090 for a realistic masked shape.

### Original optimization idea

Do the decomposition once in [`src/models/predictor.py`](/workspace/vjepa2/src/models/predictor.py),
then thread the three precomputed tensors through [`Block`](/workspace/vjepa2/src/models/utils/modules.py)
into [`RoPEAttention`](/workspace/vjepa2/src/models/utils/modules.py).

This is intentionally not a math rewrite. The attention block still uses the
same RoPE math. The only change is that it receives already-separated position
tensors instead of recomputing them every block.

### What survived after review

The precompute itself did not survive.

What stayed in the live path is the correctness cleanup around the same area:

1. predictor RoPE blocks now receive the real `grid_height` and `grid_width`,
2. non-square masked predictor inputs no longer fall back to square-grid RoPE
   decomposition,
3. `has_cls=True` with RoPE no longer crashes because prefix tokens are now
   handled explicitly in [`RoPEAttention`](/workspace/vjepa2/src/models/utils/modules.py).

### Benchmark

There are two relevant numbers here.

First, once the predictor path was corrected, the internal dynamic-vs-precompute
check said the corrected dynamic path was faster than the precomputed one:

```bash
cd /workspace/vjepa2
python - <<'PY'
import src.models.predictor as predictor_mod
...
PY
```

Measured result on the short masked-RoPE predictor check (`B=8`, `N_ctxt=64`,
`depth=4`, `fp16`, CUDA):

| benchmark | dynamic corrected path | precomputed variant | speedup | note |
| --- | ---: | ---: | ---: | ---: |
| `predictor_rope_internal` | 13.9165 ms | 14.2865 ms | 0.973x | precompute loses |

Second, the targeted current-vs-`HEAD` square benchmark also turned negative
after the correctness fixes were in place:

| benchmark | `HEAD` square check | current corrected path | speedup |
| --- | ---: | ---: | ---: |
| `predictor_rope_forward` | 12.7941 ms | 13.7104 ms | 0.928x |

### Decision

- reject the precompute optimization,
- keep the non-square RoPE fix,
- keep the `has_cls=True` RoPE crash fix,
- keep the tests that lock those correctness fixes in.

Supporting coverage:

- [`tests/test_model_block_parity.py`](/workspace/vjepa2/tests/test_model_block_parity.py)
- [`tests/test_predictor_parity.py`](/workspace/vjepa2/tests/test_predictor_parity.py)
- [`benchmarks/baseline_compare.py`](/workspace/vjepa2/benchmarks/baseline_compare.py)

## Current Fast Validation Stack

Current short validation command:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/models/test_triton_rope_kernel.py \
  tests/models/test_attention_correctness.py \
  tests/test_predictor_parity.py \
  tests/test_ac_predictor_parity.py \
  tests/test_kernel_parity.py \
  tests/models/test_ac_rope_attention.py \
  tests/test_model_block_parity.py
```

Current result:

- `34 passed`

That matters because the retained set is now:

- the previous wins that survived earlier review,
- the new stacked multi-mask `apply_masks` paths for both 2D and 1D cases,
- the predictor RoPE correctness fixes for non-square inputs and `has_cls=True`,
- and not the stale rejected `ac_predictor` cache experiment.

Those are exactly the kind of changes that are worth doing under an
"instant-test" rule because:

- they are local,
- they are easy to compare to `HEAD`,
- they have a clean parity story.

### What changed

In [`src/models/predictor.py`](/workspace/vjepa2/src/models/predictor.py):

- `predictor_pos_embed.repeat(...)` became `expand(...)`
- mask-token broadcast uses `expand(...)`
- context duplication uses `unsqueeze().expand().reshape(...)`
- reorder and reverse-reorder use `torch.gather(...)`
- the common single-mask case now avoids extra repeat/concat work entirely

### How it was validated

Parity against `HEAD`:

- [`tests/test_predictor_parity.py`](/workspace/vjepa2/tests/test_predictor_parity.py)

Fast local validation command:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/test_predictor_parity.py \
  tests/models/test_predictor.py \
  tests/models/test_ac_rope_attention.py \
  tests/test_kernel_parity.py
```

Result during this pass:

- `16 passed`

### Fast directional GPU check

I used a small forward-only `HEAD` comparison on the 3090 instead of a long
benchmark sweep:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `src predictor forward` | 3.5376 ms | 2.7554 ms | 1.284x |

This is not a final training benchmark. It is enough to keep a small local
cleanup under the current fast-iteration rule, but not enough to advertise a
large default-path predictor speedup by itself.

## Step 8: Encoder Single-Mask Fast Path

### Why this was examined

The plain encoder in [`src/models/vision_transformer.py`](/workspace/vjepa2/src/models/vision_transformer.py)
already calls [`apply_masks(...)`](/workspace/vjepa2/src/masks/utils.py), but it
still concatenated the mask list unconditionally afterward.

That is wasted work in the common case where there is exactly one mask.

### What changed

The encoder now keeps the single-mask case as:

```python
masks = masks[0] if len(masks) == 1 else torch.cat(masks, dim=0)
```

This is a tiny cleanup, but it is on the forward path and was trivial to
validate.

### Fast directional GPU check

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `masked vit_tiny forward` | 10.5897 ms | 10.4081 ms | 1.017x |

This is not a headline win, but it is positive and safe enough to keep.

## Step 9: `AttentivePooler` Query Broadcast

### Why this was examined

[`src/models/attentive_pooler.py`](/workspace/vjepa2/src/models/attentive_pooler.py)
still materialized the query tokens every forward with:

```python
q = self.query_tokens.repeat(len(x), 1, 1)
```

Those query tokens are identical across the batch, so `repeat` is unnecessary
copying.

### What changed

The forward path now uses:

```python
q = self.query_tokens.expand(len(x), -1, -1)
```

This keeps the same logical tensor shape and avoids the batch copy.

### How it was validated

Parity against `HEAD`:

- [`tests/test_model_block_parity.py`](/workspace/vjepa2/tests/test_model_block_parity.py)

Quick validation command:

```bash
cd /workspace/vjepa2
pytest -q tests/test_model_block_parity.py
```

### Fast directional GPU check

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `attentive_pooler forward` | 2.2042 ms | 1.9425 ms | 1.135x |

This is a small but real forward-path cleanup, so it stays.

## Step 10: Cached Separated RoPE Positions

### Why this was examined

The earlier RoPE cache only saved the base `arange(...)` tensor. In the no-mask
path, both RoPE attention variants still recomputed the separated frame,
height, and width position vectors every forward.

That work is deterministic for fixed `(T, H, W, grid_size)` and happens in the
hottest attention blocks in the repo.

### What changed

In [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py),
the no-mask path now caches the fully separated position vectors keyed by:

- device type and device index,
- `T`,
- `H`,
- `W`,
- `grid_size`.

For `ACRoPEAttention`, the cached tensors already include the final spatial
scaling applied to height and width positions.

### How it was validated

Fast parity set:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/test_kernel_parity.py \
  tests/models/test_ac_rope_attention.py \
  tests/test_model_block_parity.py
```

Result during this pass:

- `14 passed`

### Fast `HEAD` comparison on GPU

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `rope_attention_forward` | 5.6462 ms | 4.2448 ms | 1.330x |
| `ac_rope_attention_forward` | 10.3640 ms | 6.0950 ms | 1.700x |

This is a strong keep at the RoPE block level. It is exactly the kind of change
that fits the current optimization rule: local, parity-clean, and positive on
the real forward path that actually uses those RoPE blocks.

## Step 11: RoPE Frequency Outer Product Without `einsum`

### Why this was examined

After caching inverse frequencies and separated positions, the remaining hot
primitive in the RoPE path was the frequency outer-product itself:

```python
freq = torch.einsum("..., f -> ... f", pos, omega)
```

That is clean code, but it is still more machinery than the operation really
needs. In this case, the math is just a broadcast multiply.

### What changed

In [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py),
the RoPE helper now uses:

```python
freq = pos.unsqueeze(-1) * omega
```

Nothing else in the helper changed.

### Fast isolated check

I first tested the primitive by itself before touching the module:

| position shape | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `(1024,)` | 0.6764 ms | 0.6270 ms | 1.079x |
| `(1, 1, 1024)` | 0.6664 ms | 0.6277 ms | 1.062x |

That was enough to justify patching the live path and then checking the real
attention blocks.

### How it was validated

Fast parity set:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/test_kernel_parity.py \
  tests/models/test_ac_rope_attention.py \
  tests/test_model_block_parity.py
```

Result during this pass:

- `14 passed`

### Updated `HEAD` comparison on GPU

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `rotate_queries_or_keys` | 0.7305 ms | 0.4913 ms | 1.487x |
| `rope_attention_forward` | 5.7669 ms | 4.2153 ms | 1.368x |
| `ac_rope_attention_forward` | 10.3764 ms | 6.0455 ms | 1.716x |

This one stays. It is a hot primitive, it is parity-clean, and the improvement
shows up clearly in both RoPE-heavy block benchmarks.

## Step 12: Preserve The RoPE Compatibility Quirk With `cat`, Not `repeat`

### Why this was examined

After removing `einsum`, the next remaining RoPE hot spot was the compatibility
duplication step:

```python
emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)
```

This repo intentionally preserves the pretrained compatibility quirk where the
frequency vector is duplicated as `[f0, f1, ..., f0, f1, ...]`, not
interleaved pairwise.

The key observation was that the quirk can be preserved without using
`repeat(...)`. A simple concatenation of the half-vectors produces the same
layout:

```python
half = emb_sin.squeeze(-1)
emb_sin = torch.cat([half, half], dim=-1)
```

### Fast isolated check

Before patching the module, I tested the primitive directly:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `rotate_queries_or_keys` helper | 0.6604 ms | 0.5575 ms | 1.185x |

That was enough to justify pushing the change into the real path.

### How it was validated

Fast parity set:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/test_kernel_parity.py \
  tests/models/test_ac_rope_attention.py \
  tests/test_model_block_parity.py
```

Result during this pass:

- `14 passed`

### Updated `HEAD` comparison on GPU

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `rotate_queries_or_keys` | 0.7462 ms | 0.4116 ms | 1.813x |
| `rope_attention_forward` | 6.1455 ms | 3.8386 ms | 1.601x |
| `ac_rope_attention_forward` | 11.1655 ms | 5.3549 ms | 2.085x |

This stays. It keeps the pretrained-compatible behavior, improves the hot
primitive again, and pushes both RoPE block benchmarks materially farther ahead
of `HEAD`. That still does not make it a standalone full-model win claim; it is
one contributing RoPE-side cleanup inside a stack of related changes.

## Step 13: Triton RoPE Fast Path Experiment (Rejected)

### Why this was examined

After the latest RoPE helper cleanups, the profile still showed the remaining
cost concentrated in the actual rotate primitive:

- trigonometric work,
- elementwise multiplies,
- repeated concatenation/materialization.

That is exactly the kind of pattern where a fused kernel can win after the easy
PyTorch cleanups have already been taken.

### Prototype outcome

I prototyped an optional Triton kernel out of tree first and only briefly wired
it into the repo after the forward-only measurements looked very good.

The idea was straightforward:

- fuse the pair rotation,
- fuse the sine/cosine application,
- skip the intermediate RoPE materializations.

### How the kernel works

For each `(b, h, n, d)` element, the kernel:

1. loads `x[b, h, n, d]`,
2. loads the paired element `x[b, h, n, d ^ 1]`,
3. forms the rotated partner value (`-pair` for even lanes, `pair` for odd),
4. loads the matching RoPE frequency using the current pretrained-compatible
   duplication layout,
5. computes the final rotated output directly.

That means the Triton path skips:

- building `freq` as a separate tensor,
- materializing duplicated sine/cosine tensors,
- materializing the rotated `y` tensor.

### Fast isolated check against the current optimized PyTorch path

Before wiring it into the repo, I compared the Triton helper directly to the
already-optimized PyTorch helper:

| dtype | baseline | Triton | speedup |
| --- | ---: | ---: | ---: |
| `fp16` | 0.3652 ms | 0.2865 ms | 1.275x |
| `fp32` | 0.5162 ms | 0.2799 ms | 1.845x |

That was enough to justify integrating it behind the existing API.

### Forward-only measurements

Fast parity set:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/test_kernel_parity.py \
  tests/models/test_ac_rope_attention.py \
  tests/test_model_block_parity.py
```

Those numbers were attractive enough that I tried the integration.

### Updated `HEAD` comparison on GPU

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `rotate_queries_or_keys` | 0.7092 ms | 0.1957 ms | 3.624x |
| `rope_attention_forward` | 5.7322 ms | 1.4318 ms | 4.004x |
| `ac_rope_attention_forward` | 10.2854 ms | 2.5210 ms | 4.080x |

### Why it was rejected

The forward-only numbers were not enough.

When I checked gradient flow directly, the Triton path failed the real
requirement:

- the output tensor did **not** require grad,
- backward failed immediately.

In other words, the prototype was a forward-only fast path with no usable
autograd story.

That is a hard reject for this repo because these paths are used in training.

I also ran small full-model default-path checks on the real `use_rope=True`,
`use_sdpa=True` paths before the revert:

| benchmark | baseline | Triton experiment | speedup |
| --- | ---: | ---: | ---: |
| encoder, unmasked | 11.1085 ms | 11.3893 ms | 0.975x |
| encoder, masked | 25.5810 ms | 25.6768 ms | 0.996x |
| predictor | 24.0408 ms | 24.4017 ms | 0.985x |
| action-conditioned predictor | 19.4498 ms | 18.7881 ms | 1.035x |

So even ignoring the autograd failure, the default plain encoder/predictor path
was not a clean win in these short model-level checks.

Conclusion:

- reverted from the codebase,
- kept only as a documented experiment,
- if revisited later, it needs a real autograd-safe design, not just a fast
  forward kernel.

## Step 14: Fuse `q` And `k` Rotation Calls

### Why this was examined

After rejecting the Triton prototype, the next best RoPE target was still in the
same area, but now under the constraint that it had to stay fully autograd-safe
and PyTorch-native.

The attention code still did this pattern repeatedly:

```python
qd = rotate_queries_or_keys(q_slice, pos)
kd = rotate_queries_or_keys(k_slice, pos)
```

for depth, height, and width segments.

That means the same position/frequency work gets repeated once for `q` and once
for `k`, even though both use the same `pos`.

### Optimization idea

Concatenate `q` and `k` along the leading dimension, rotate once, then split
them back:

```python
qk = torch.cat([q, k], dim=0)
qk = rotate_queries_or_keys(qk, pos)
q, k = qk[:q.size(0)], qk[q.size(0):]
```

For masked paths, the same trick works as long as the position tensor is
duplicated along the matching leading dimension too.

### Fast isolated check

Before patching the modules, I benchmarked the primitive directly:

| case | baseline | fused `qk` rotate | speedup |
| --- | ---: | ---: | ---: |
| `B=8, H=6, N=64, D=20` | 0.7759 ms | 0.4426 ms | 1.753x |
| `B=8, H=6, N=256, D=20` | 0.7806 ms | 0.4626 ms | 1.687x |
| `B=8, H=16, N=1024, D=20` | 0.7792 ms | 0.4471 ms | 1.743x |

That was strong enough to justify patching the live path.

### What changed

In [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py):

- added a small `rotate_query_key_pair(...)` helper,
- used it in `RoPEAttention`,
- used it in `ACRoPEAttention`,
- used it in the action-token depth-rotation path as well.

### How it was validated

Fast parity set:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/test_kernel_parity.py \
  tests/models/test_ac_rope_attention.py \
  tests/test_model_block_parity.py \
  tests/models/test_models.py
```

Result during this pass:

- `22 passed`

Broader quick validation:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/models/test_models.py \
  tests/test_predictor_parity.py \
  tests/models/test_predictor.py \
  tests/test_kernel_parity.py \
  tests/models/test_ac_rope_attention.py \
  tests/test_model_block_parity.py
```

Result during this pass:

- `27 passed`

### Updated measurements against `HEAD`

Default SDPA block path:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `RoPEAttention_sdpa_default` | 2.0765 ms | 2.0580 ms | 1.009x |
| `ACRoPEAttention_sdpa_default` | 3.7601 ms | 3.5005 ms | 1.074x |

Small full-model checks:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| encoder, unmasked | 16.6889 ms | 16.7961 ms | 0.994x |
| encoder, masked | 19.1945 ms | 19.2326 ms | 0.998x |
| predictor | 19.9052 ms | 19.8644 ms | 1.002x |
| action-conditioned predictor | 26.4677 ms | 26.1636 ms | 1.012x |

Interpretation:

- this is a real, safe win at the RoPE block level,
- it is close to neutral on the plain full encoder in these short checks,
- it remains positive on the action-conditioned path,
- unlike the Triton prototype, it preserves normal training behavior.

## Current Retained Wins

These are the improvements I would currently claim as real under the current
instant-test standard.

That does not mean every row below is a headline default-path model speedup.
Some are:

- clear main-path wins,
- clear block-level wins with near-neutral short full-model effect,
- or correctness/local-cleanup changes that are still worth keeping.

| area | file | result |
| --- | --- | --- |
| Action causal mask build | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py) | large measured win |
| Batch repeat-interleave helper | [`src/utils/tensors.py`](/workspace/vjepa2/src/utils/tensors.py) | clear measured win |
| Mask gather index expansion | [`src/masks/utils.py`](/workspace/vjepa2/src/masks/utils.py) | moderate measured win |
| Batched action-token path in `ACRoPEAttention` | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py) | strong measured win in baseline-vs-`HEAD` block benchmark |
| RoPE-side helper cleanups: inverse-frequency caching, separated-position caching, broadcast outer-product, and compatibility duplication via `cat` | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py), [`app/vjepa_2_1/models/utils/modules.py`](/workspace/vjepa2/app/vjepa_2_1/models/utils/modules.py) | clear RoPE block wins; individual helper wins should be read as contributing factors, not separate full-model claims |
| Fused `qk` RoPE rotation | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py) | safe RoPE block-level win; short full-model checks are near neutral outside the action-conditioned path |
| Predictor broadcast/gather cleanup | [`src/models/predictor.py`](/workspace/vjepa2/src/models/predictor.py) | safe local cleanup with a positive short directional check; default-path full-model impact is small |
| Encoder single-mask fast path | [`src/models/vision_transformer.py`](/workspace/vjepa2/src/models/vision_transformer.py) | small positive masked-forward win |
| `AttentivePooler` query broadcast | [`src/models/attentive_pooler.py`](/workspace/vjepa2/src/models/attentive_pooler.py) | small positive forward-path win |
| SDPA wrapper + eval correctness | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py), [`app/vjepa_2_1/models/utils/modules.py`](/workspace/vjepa2/app/vjepa_2_1/models/utils/modules.py) | correctness fix, small/neutral perf effect |

## Test and Benchmark Commands

### Full retained validation

```bash
cd /workspace/vjepa2
pytest -q \
  tests/models/test_kernel_speedups.py \
  tests/models/test_sdpa_attention.py \
  tests/models/test_attention_kernels.py \
  tests/test_kernel_parity.py \
  tests/test_model_block_parity.py \
  tests/models/test_models.py \
  tests/support/test_kernel_benchmark.py
```

### Utility and block baseline benchmark

```bash
cd /workspace/vjepa2
python benchmarks/baseline_compare.py
```

## What I Would Do Next

The next pass should not be "write 100 random kernels".

The next good targets are:

1. Full model-level forward/backward benchmarks on real training shapes.
2. Predictor-side masking and gather paths.
3. End-to-end profiling to find time spent outside attention.
4. Only then: consider Triton or custom kernels where the profile proves they are worth it.

The standard should stay the same:

- find hotspot,
- preserve semantics,
- compare against baseline,
- keep only what wins.

## Step 15: Manual Kernel Review And Safety Pass

I revisited the Triton RoPE work with a stricter rule:

- forward-only wins are fine,
- training-path wins must prove backward correctness,
- if a manual kernel fails parity, it is removed from the live path immediately.

Two important outcomes came from that pass:

1. The live Triton autograd path was disabled.
2. The retained Triton kernel is now forward-only for safe no-grad execution.

Why:

- the custom backward did not match the repo's compatibility-preserving RoPE
  behavior,
- gradient parity failed,
- keeping it enabled for training would have been a correctness bug.

Retained measurement for the safe forward-only kernel on `[8, 16, 1024, 64]`,
`fp16`, CUDA:

| kernel | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `rotate_queries_or_keys` forward | 0.5608 ms | 0.1989 ms | 2.82x |

Validation:

- raw Triton forward checked against a pure PyTorch reference,
- live gradient path checked through the exported helper,
- result: forward win retained, live gradients exact because training falls back
  to PyTorch.

## Step 16: Rejected Manual Kernel Experiments

Three manual-kernel ideas were tried and rejected in this pass:

1. Triton autograd for the RoPE rotate primitive.
2. A fused Triton multi-axis `q/k` RoPE kernel.
3. A Triton extension for batched masked positions.

All three looked attractive in principle.
All three were rejected for the same reason:

- parity first,
- speed second.

Specific outcomes:

- the autograd path failed gradient parity,
- the fused multi-axis kernel failed forward parity on realistic shapes,
- the masked batched-position extension also failed parity even though its
  microbenchmark looked strong.

This is exactly why the keep/reject loop exists.

## Step 17: Triton `apply_masks` Kernel (Rejected)

I tried a fused Triton gather kernel for [`src/masks/utils.py`](/workspace/vjepa2/src/masks/utils.py).

Design:

- manual Triton forward for packed multi-mask gathers,
- exact backward through scatter-add,
- narrow CUDA-only gate,
- parity-first rollout.

Results:

Primitive benchmark on `[8, 1024, 384]`, `4` mask groups, `fp16`, CUDA:

| kernel | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `apply_masks` | 0.2958 ms | 0.2611 ms | 1.13x |

Backward parity:

- exact match on the short CUDA check.

Caller-level checks:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| masked image encoder | 52.0714 ms | 52.1849 ms | 0.998x |
| masked video encoder | 40.1243 ms | 40.2205 ms | 0.998x |

Decision:

- reject,
- revert from the live path,
- keep the measurement in the notes.

This is a textbook case of a primitive win that does not matter once the whole
caller is measured.

## Step 18: Fused `q/k` RoPE Pair Kernel (Retained)

The next clean target was the repeated `rotate_query_key_pair` path in
[`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py).
This path already mattered because it is used by both standard RoPE attention
and the action-conditioned RoPE blocks.

Before this change, the pair helper did two things:

1. concatenated `q` and `k`,
2. called the single-tensor rotate helper on the temporary packed tensor.

That was correct, but it still paid for a temporary pack and it still rotated
the two tensors as if they were independent rows.

The new Triton path in
[`src/models/utils/triton_kernels.py`](/workspace/vjepa2/src/models/utils/triton_kernels.py)
does the pair directly:

- one kernel launch,
- one position load,
- one sin/cos evaluation,
- two outputs written in the same pass.

### Primitive benchmark

Benchmark shape: `[8, 16, 4096, 24]`, `fp16`, CUDA.

| kernel | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `rotate_query_key_pair` | 1.3617 ms | 0.2182 ms | 6.24x |

That is the kind of result that clears the bar immediately.

### Caller-level benchmark

The real check is the module-level inference path under `no_grad`, where this
pair helper actually gets used.

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `ac_rope_attention_forward` | 2.0785 ms | 1.9283 ms | 1.08x |
| `rope_attention_forward` | 1.1694 ms | 1.2025 ms | 0.97x |

Decision:

- keep the pair kernel,
- keep the action-conditioned RoPE win,
- reject the plain RoPE caller win on this shape because it did not improve.

The result is still useful because the pair kernel is now a real retained win in
the exact hot path that matters most for action-conditioned attention.

### Validation cleanup

The focused validation after landing this kernel is currently:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/test_kernel_parity.py \
  tests/test_model_block_parity.py \
  tests/test_predictor_parity.py \
  tests/models/test_ac_rope_attention.py \
  tests/models/test_attention_correctness.py
```

Result:

- `35 passed`

The small predictor positional-add cleanup also stays in place so `no_grad`
evaluation does not rely on in-place writes into expanded mask-token views.

## Step 19: Two More Rejections That Looked Plausible

This pass also tested two fresh ideas that are worth recording precisely because
they did not survive the caller-level filter.

### Rejected: contiguous patch-embed output

Hypothesis:

- [`PatchEmbed`](/workspace/vjepa2/src/models/utils/patch_embed.py) and
  [`PatchEmbed3D`](/workspace/vjepa2/src/models/utils/patch_embed.py) return
  `flatten(...).transpose(1, 2)`,
- that output is non-contiguous,
- making it contiguous once at the encoder boundary might help the downstream
  blocks more than it costs at patch embed time.

What the short checks showed:

- patch-embed itself got slower once the explicit copy was added,
- encoder-level unmasked forward got faster,
- masked encoder forward regressed.

Measured image encoder numbers (`img_size=128`, `depth=4`, `fp16`, CUDA):

| benchmark | baseline | contiguous variant | speedup |
| --- | ---: | ---: | ---: |
| image unmasked | 3.8271 ms | 3.7288 ms | 1.03x |
| image masked | 3.7480 ms | 3.7693 ms | 0.99x |

Measured video encoder numbers (`num_frames=8`, `depth=4`, `fp16`, CUDA):

| benchmark | baseline | contiguous variant | speedup |
| --- | ---: | ---: | ---: |
| video unmasked | 3.8688 ms | 3.5719 ms | 1.08x |
| video masked | 3.6584 ms | 3.7410 ms | 0.98x |

Backward parity was exact on the short CUDA probe, so correctness was not the
problem. The problem was that the main masked path did not improve.

Decision:

- reject,
- keep the measurement,
- do not spend more time on this unless the target workload shifts toward
  unmasked inference.

### Rejected: direct mask-token expansion in predictor

Hypothesis:

- the predictor mask token is identical at every patch position,
- so instead of expanding to `[B, num_patches, D]` and then gathering with
  [`apply_masks(...)`](/workspace/vjepa2/src/masks/utils.py), expand directly to
  the target masked shape.

This also required an out-of-place positional add to avoid aliasing when the
target tensor came from `expand`.

Benchmark script:

- [`benchmarks/predictor_mask_tokens.py`](/workspace/vjepa2/benchmarks/predictor_mask_tokens.py)

Measured result on the short CUDA predictor check:

| benchmark | baseline | direct-expand variant | speedup |
| --- | ---: | ---: | ---: |
| predictor mask-token path | 6.6483 ms | 6.7706 ms | 0.98x |

Decision:

- reject,
- keep the benchmark artifact,
- leave the live predictor path unchanged.

## Step 20: App Predictor Output Reorder Cleanup (Rejected)

The app-side predictor in
[`app/vjepa_2_1/models/predictor.py`](/workspace/vjepa2/app/vjepa_2_1/models/predictor.py)
was still paying for a full inverse permutation on the hot `return_all_tokens=False`
path, even though the mask pipeline already provides sorted mask indices.

The experiment was narrow:

- compute sorted target positions directly with `torch.searchsorted(...)`,
- gather only the target slice when `return_all_tokens=False`,
- keep the inverse-permutation fallback for the `return_all_tokens=True` path.

This was only valid for the sorted-mask contract, so the benchmark and parity
test used sorted masks to match the real producer.

### Parity

Targeted `HEAD` parity:

```bash
cd /workspace/vjepa2
pytest -q tests/test_app_predictor_parity.py tests/test_app_predictor_output_reorder.py
```

Result:

- `3 passed`

The coverage lives in:

- [`tests/test_app_predictor_parity.py`](/workspace/vjepa2/tests/test_app_predictor_parity.py)
- [`tests/test_app_predictor_output_reorder.py`](/workspace/vjepa2/tests/test_app_predictor_output_reorder.py)

### Benchmark

Benchmark CLI:

```bash
cd /workspace/vjepa2
python benchmarks/app_predictor_compare.py
```

One-off CUDA run:

| benchmark | baseline | optimized | speedup | improvement |
| --- | ---: | ---: | ---: | ---: |
| `app_predictor_forward` | 14.9866 ms | 14.5144 ms | 1.03x | 3.15% |

Supporting benchmark:

- [`benchmarks/app_predictor_compare.py`](/workspace/vjepa2/benchmarks/app_predictor_compare.py)

Decision:

- reject,
- repeated CUDA measurements on the sorted-mask path averaged about `-0.45%`
  over 4 runs, with 3 of 4 runs slower,
- do not keep live `predictor.py` changes.

## Step 21: Historical RoPE Pair Work

There was an earlier app-side RoPE pair-path experiment in
[`app/vjepa_2_1/models/utils/modules.py`](/workspace/vjepa2/app/vjepa_2_1/models/utils/modules.py).
It helped the primitive `rotate_query_key_pair(...)` microbench, but that path is
not a retained app predictor win.

Keep the benchmark record if you want the history, but do not confuse it with
the rejected app predictor output-reorder change above.

## Step 22: Triton RoPE Launch Tuning

The manual Triton RoPE kernels were already correct, but the 3090 profile showed
that the launch configuration was leaving performance on the table.

The retained change is small and explicit:

- keep the same Triton math,
- lower the launch width for `rotate_queries_or_keys` to `num_warps=2`,
- keep the paired kernel on Triton, but bias it toward a wider tile with
  `block_d=128` and `num_warps=2`,
- leave all backward behavior untouched because these paths are still
  forward-only / no-grad scoped.

### Parity

The tuned launch config still matches the PyTorch reference:

```bash
cd /workspace/vjepa2
pytest -q tests/models/test_triton_rope_kernel.py tests/test_kernel_parity.py -k 'rotate_queries_or_keys or rotate_query_key_pair or triton_rope'
```

Result:

- `6 passed`

### Benchmark

Current path vs PyTorch baseline on the same 3090 shape used elsewhere in the
repo:

```bash
cd /workspace/vjepa2
python benchmarks/kernel_speedups.py
```

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `rotate_queries_or_keys` | 1.0148 ms | 0.9010 ms | 11.22% |
| `rotate_query_key_pair` | 1.4236 ms | 0.7710 ms | 45.84% |

Launch-tuning comparison against the previous Triton config, keeping the math
identical and only changing `BLOCK_D` / `num_warps`:

| benchmark | old launch config | tuned config | improvement |
| --- | ---: | ---: | ---: |
| `rotate_queries_or_keys` | 0.9450 ms | 0.4668 ms | 50.57% |
| `rotate_query_key_pair` | 1.9178 ms | 1.3901 ms | 27.53% |

That is the right kind of Triton win:

- no functionality change,
- exact parity still holds,
- the 3090 likes the smaller warp count,
- and the paired kernel also benefits from a larger tile.

## Article Version

For a cleaned-up tutorial version of this work, see
[`OPTIMIZATION_ARTICLE.md`](/workspace/vjepa2/docs/optimization/OPTIMIZATION_ARTICLE.md).

That file keeps the lessons and results organized by topic.
This file stays the chronological engineering record.

## Step 23: Training-Safe Triton RoPE

The next real manual-kernel step was to stop treating Triton RoPE as
forward-only.

The retained change in
[`src/models/utils/triton_kernels.py`](/workspace/vjepa2/src/models/utils/triton_kernels.py)
adds a custom autograd wrapper for the existing Triton rotate kernel.
Backward is not new math. RoPE is an orthonormal rotation, so the input
gradient is the same rotation applied with the negated position angle.

That lets
[`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py)
use the Triton path even when gradients are enabled:

- forward stays on the same proven Triton kernel,
- backward uses the inverse rotation through the same Triton primitive,
- no-grad inference still uses the plain forward Triton path,
- pair rotation in training picks this up automatically through
  `rotate_queries_or_keys(...)`.

### Parity

Quick validation:

```bash
cd /workspace/vjepa2
pytest -q \
  tests/models/test_triton_rope_kernel.py \
  tests/test_kernel_parity.py \
  tests/models/test_attention_correctness.py \
  tests/models/test_ac_rope_attention.py \
  tests/test_app_predictor_parity.py \
  tests/test_app_predictor_output_reorder.py \
  tests/test_model_block_parity.py
```

Result:

- `37 passed`

New backward coverage lives in:

- [`tests/models/test_triton_rope_kernel.py`](/workspace/vjepa2/tests/models/test_triton_rope_kernel.py)

### Benchmark

Reproducible training-mode `HEAD` comparisons now live in
[`benchmarks/baseline_compare.py`](/workspace/vjepa2/benchmarks/baseline_compare.py).

Command:

```bash
cd /workspace/vjepa2
python - <<'PY'
from benchmarks.baseline_compare import load_module_from_head, compare_rope_attention_train, compare_ac_rope_attention_train, ROOT
baseline_modules = load_module_from_head(ROOT, 'src/models/utils/modules.py', 'baseline_src_models_utils_modules_train_check')
for row in [compare_rope_attention_train(baseline_modules), compare_ac_rope_attention_train(baseline_modules)]:
    row['speedup_pct'] = ((row['baseline_ms'] - row['optimized_ms']) / row['baseline_ms']) * 100.0
    print(row)
PY
```

Fresh CUDA result:

| benchmark | baseline | optimized | speedup | improvement |
| --- | ---: | ---: | ---: | ---: |
| `rope_attention_train` | 9.3340 ms | 7.5418 ms | 1.24x | 19.20% |
| `ac_rope_attention_train` | 14.1698 ms | 11.6688 ms | 1.21x | 17.65% |

Supporting primitive check from the same pass:

| benchmark | baseline | optimized | speedup | improvement |
| --- | ---: | ---: | ---: | ---: |
| `rotate_queries_or_keys_train` | 2.1162 ms | 0.9759 ms | 2.17x | 53.88% |

### App Predictor Rerun

The app predictor output-reorder experiment did not survive repeated reruns:

```bash
cd /workspace/vjepa2
python benchmarks/app_predictor_compare.py
```

| benchmark | baseline | optimized | speedup | improvement |
| --- | ---: | ---: | ---: | ---: |
| one-off sorted-mask run | 14.6492 ms | 14.2557 ms | 1.03x | 2.69% |
| repeated sorted-mask mean | 14.6607 ms | 14.7262 ms | 1.00x | -0.45% |

Decision:

- keep the training-safe Triton RoPE path,
- reject the app predictor sorted-target extraction path as noise,
- and treat the old "forward-only Triton" note as historical context, not the
  final state of the live path.

## Step 24: AC Predictor Glue Sweep (Rejected)

I also checked the action-conditioned predictor in
[`src/models/ac_predictor.py`](/workspace/vjepa2/src/models/ac_predictor.py).
The obvious glue candidates were the causal mask path and the conditioning-token
pack/unpack path.

I did not keep any change.

### What I Checked

- causal-mask handling,
- conditioning-token packing,
- conditioning-token unpacking,
- the existing parity helper in
  [`tests/test_ac_predictor_parity.py`](/workspace/vjepa2/tests/test_ac_predictor_parity.py).

### Results

The repeated small-shape default path was only marginally positive:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `ac_predictor_forward` | 5.7273 ms | 5.5867 ms | 1.82% |

But the extrinsics branch regressed:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `ac_predictor_forward_extrinsics` | 5.8428 ms | 6.1466 ms | -5.03% |

A larger one-off benchmark was also negative:

| benchmark | baseline | optimized | speedup |
| --- | ---: | ---: | ---: |
| `ac_predictor_forward` | 10.3075 ms | 10.3602 ms | -0.51% |

### Decision

- reject,
- keep `src/models/ac_predictor.py` at `HEAD`,
- do not promote AC predictor glue changes unless they survive repeated caller-level benchmarking on the main path.

## Step 25: Prototype Kernels That Did Not Graduate

I also checked the remaining manual-kernel prototypes that had accumulated in
the tree.

### Separable RoPE Prototype

The prototype in
[`src/models/utils/fused_qkv_rope_kernel.py`](/workspace/vjepa2/src/models/utils/fused_qkv_rope_kernel.py)
tried to fuse the three `d/h/w` RoPE slices into one Triton pass.

That is the right direction conceptually, but the current prototype did not
survive the first real gate:

- it needed immediate compile fixes just to run on the 3090,
- and after that it still failed parity badly against the existing slice-wise
  reference path.

Decision:

- reject the current prototype,
- keep the idea in the notebook,
- do not promote any fused separable-RoPE kernel until it matches the existing
  `d/h/w` split path exactly.

### GQA / Sparse Attention Prototypes

The stray files

- [`src/models/utils/gqa_kernel.py`](/workspace/vjepa2/src/models/utils/gqa_kernel.py)
- [`src/models/utils/sparse_attention_kernel.py`](/workspace/vjepa2/src/models/utils/sparse_attention_kernel.py)
- [`src/models/utils/temporal_aggregation_kernel.py`](/workspace/vjepa2/src/models/utils/temporal_aggregation_kernel.py)

are not wired into the live V-JEPA 2 attention path.

So even before benchmarking, they fail the main-path filter for this pass.

Decision:

- treat them as off-path prototypes,
- do not count them as optimizations for the current model,
- remove them from the active working set unless and until the architecture
  actually needs them.
