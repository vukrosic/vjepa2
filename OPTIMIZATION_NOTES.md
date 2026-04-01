# V-JEPA 2 Optimization Notes

This file is a running engineering note for the optimization work in this repo.

It is intentionally written like a tutorial blog post, not a terse changelog.
The goal is to make each speedup reproducible, explain why it is safe, and show
how to measure it against a baseline before keeping it.

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

This is not a final training benchmark, but it is enough to keep the change in
the codebase under the current fast-iteration rule.

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

This is a strong keep. It is exactly the kind of change that fits the current
optimization rule: local, parity-clean, and positive on the real forward path.

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
of `HEAD`.

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

These are the improvements I would currently claim as real:

| area | file | result |
| --- | --- | --- |
| Action causal mask build | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py) | large measured win |
| RoPE inverse-frequency caching | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py), [`app/vjepa_2_1/models/utils/modules.py`](/workspace/vjepa2/app/vjepa_2_1/models/utils/modules.py) | clear measured win on RoPE paths |
| RoPE cached positions | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py), [`app/vjepa_2_1/models/utils/modules.py`](/workspace/vjepa2/app/vjepa_2_1/models/utils/modules.py) | reduces repeated `arange` overhead |
| Batch repeat-interleave helper | [`src/utils/tensors.py`](/workspace/vjepa2/src/utils/tensors.py) | clear measured win |
| Mask gather index expansion | [`src/masks/utils.py`](/workspace/vjepa2/src/masks/utils.py) | moderate measured win |
| Batched action-token path in `ACRoPEAttention` | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py) | strong measured win in baseline-vs-`HEAD` block benchmark |
| RoPE frequency outer-product broadcast | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py) | clear primitive win and positive RoPE block impact |
| RoPE compatibility duplication via `cat` | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py) | strong primitive win and clear RoPE block impact |
| Fused `qk` RoPE rotation | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py) | safe block-level win, mild positive/neutral full-model effect |
| Predictor broadcast/gather cleanup | [`src/models/predictor.py`](/workspace/vjepa2/src/models/predictor.py) | quick directional win, parity-tested |
| Encoder single-mask fast path | [`src/models/vision_transformer.py`](/workspace/vjepa2/src/models/vision_transformer.py) | small positive masked-forward win |
| `AttentivePooler` query broadcast | [`src/models/attentive_pooler.py`](/workspace/vjepa2/src/models/attentive_pooler.py) | small positive forward-path win |
| Cached separated RoPE positions | [`src/models/utils/modules.py`](/workspace/vjepa2/src/models/utils/modules.py) | strong measured win in both RoPE block benchmarks |
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

## Article Version

For a cleaned-up tutorial version of this work, see
[`OPTIMIZATION_ARTICLE.md`](/workspace/vjepa2/OPTIMIZATION_ARTICLE.md).

That file keeps the lessons and results organized by topic.
This file stays the chronological engineering record.
