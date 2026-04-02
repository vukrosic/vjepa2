---
title: "Improving The V-JEPA 2 Baseline"
date: "2026-04-02"
description: "A technical article on the final retained V-JEPA 2 baseline improvements: training-safe Triton RoPE, batched action-conditioned attention, predictor and masking-path cleanup, and the measured end-to-end result."
readTime: "10 min read"
tags: ["V-JEPA 2", "Triton", "Kernel Optimization", "Video Models", "RTX 3090", "Meta"]
sortOrder: 0
---

V-JEPA 2 is Meta's self-supervised video model for learning visual world representations from masked video prediction. In practice, the baseline has two kinds of work:

- large dense tensor math, mostly in attention and MLP blocks,
- a lot of surrounding code that prepares masks, rotates RoPE features, repeats tokens, gathers masked subsets, and reshapes data for the predictor.

The dense math was already in decent shape. PyTorch SDPA was doing the heavy attention work well. The meaningful improvements came from the code around that dense math: making RoPE faster without breaking training, batching action-conditioned attention work, removing repeated tensor assembly, and caching deterministic helpers.

This article describes the final retained baseline changes only: the code that stayed because it was faster, correct, and worth keeping.

## End-To-End Result

The first question is the right one: how much faster is the whole thing after the final retained changes?

I ran a short end-to-end/default-path check on the final baseline. The answer is: the overall model moves only a little.

| Benchmark | Baseline | Final | Speedup |
|:---|---:|---:|---:|
| encoder, unmasked | 16.6889 ms | 16.7961 ms | 0.994x |
| encoder, masked | 19.1945 ms | 19.2326 ms | 0.998x |
| predictor | 19.9052 ms | 19.8644 ms | 1.002x |
| action-conditioned predictor | 26.4677 ms | 26.1636 ms | 1.012x |

So the honest top-line result is:

- the baseline code is meaningfully better in several hot subpaths,
- the final default end-to-end speedup is modest,
- and the reason is simple: the model is still dominated by already-efficient dense compute, while the retained wins mostly remove overhead around it.

That does not make the improvements unimportant. It just means the right way to read them is as real code-path wins, not as a giant whole-model throughput jump.

## Where The Baseline Actually Improved

The retained improvements fall into four groups:

1. RoPE and attention-path code that survived real training use.
2. Action-conditioned attention code that stopped doing unnecessary per-token work.
3. Predictor and masking-path tensor plumbing.
4. Small deterministic helper and broadcast fixes.

The table below is the complete retained set.

## Full Table Of Final Retained Improvements

| Area | Old code | Final code | File | Final measured result | Speedup | Scope |
|:---|:---|:---|:---|---:|---:|:---|
| Triton RoPE, training path | slower baseline RoPE rotation path | training-safe Triton RoPE helpers kept on the live path | `src/models/utils/triton_kernels.py` | `rope_attention_train: 9.3340 ms -> 7.5418 ms` | **1.24x** | training |
| Triton RoPE, AC training path | same baseline rotation path on the action-conditioned branch | same training-safe Triton path retained on AC RoPE | `src/models/utils/triton_kernels.py` | `ac_rope_attention_train: 14.1698 ms -> 11.6688 ms` | **1.21x** | training |
| Fused `q/k` RoPE primitive | separate slower rotation work | fused Triton `q/k` rotate primitive used by the retained path | `src/models/utils/triton_kernels.py` | `rotate_query_key_pair: 1.3617 ms -> 0.2182 ms` | **6.24x** | primitive |
| Action-conditioned RoPE attention | loop over action tokens and repeat QKV and RoPE setup | flatten action-token work into one batched path | `src/models/utils/modules.py` | `ac_rope_attention_forward: 10.0614 ms -> 6.2051 ms` | **1.62x** | caller/block |
| Plain RoPE attention | slower helper path | retained RoPE helper cleanup and fused rotation path | `src/models/utils/modules.py` | `rope_attention_forward: 5.3331 ms -> 4.3442 ms` | **1.23x** | caller/block |
| Predictor tensor path | repeated copies and Python-side reorder work | broadcast/gather cleanup | `src/models/predictor.py` | `src predictor forward: 3.5376 ms -> 2.7554 ms` | **1.28x** | predictor path |
| `apply_masks`, 1D same-shape case | one gather path per mask group | stacked masks handled with one batched gather | `src/masks/utils.py` | `0.2736 ms -> 0.1323 ms` | **2.07x** | live helper |
| `apply_masks`, 2D same-shape case | one gather path per mask group | stacked masks handled with one batched gather | `src/masks/utils.py` | `0.2104 ms -> 0.1096 ms` | **1.92x** | live helper |
| `repeat_interleave_batch` | nested Python `cat()` | reshape, expand, reshape | `src/utils/tensors.py` | `186.69 ms -> 96.76 ms` | **1.93x** | live helper |
| Action block causal mask build | rebuild the same block mask every call | cached mask construction with clone on use | `src/models/utils/modules.py` | `24.80 ms -> 0.78 ms` | **31.79x** | live helper |
| `AttentivePooler` query tokens | `repeat` shared queries across batch | `expand` shared queries across batch | `src/models/attentive_pooler.py` | `2.2042 ms -> 1.9425 ms` | **1.14x** | forward block |
| Encoder single-mask case | concatenate even when one mask already exists | keep the one-mask case on the simple path | `src/models/vision_transformer.py` | `10.5897 ms -> 10.4081 ms` | **1.017x** | forward block |
| SDPA eval behavior | dropout could still reach SDPA in eval mode | gate SDPA dropout on `self.training` | `src/models/utils/modules.py` | no throughput claim | correctness | safety |
| Half-precision RoPE behavior | cached positions could take the wrong dtype path | keep cached positions aligned with the active dtype path | `src/models/utils/modules.py` | no throughput claim | correctness | safety |

The rows are not additive. Several of them help the same path, especially the RoPE rows. The point of the table is to show exactly what stayed in the final baseline.

## 1. Triton RoPE That Was Worth Keeping

The most interesting retained kernel work is the RoPE path.

What matters here is not just that a Triton primitive was fast. What matters is that the final RoPE path survived the stricter bar:

- it improved the primitive,
- it improved the caller,
- and it improved the training path.

That is why it stayed.

Final retained RoPE results:

| Benchmark | Baseline | Final | Speedup |
|:---|---:|---:|---:|
| `rotate_query_key_pair` | 1.3617 ms | 0.2182 ms | **6.24x** |
| `rope_attention_forward` | 5.3331 ms | 4.3442 ms | **1.23x** |
| `ac_rope_attention_forward` | 10.0614 ms | 6.2051 ms | **1.62x** |
| `rope_attention_train` | 9.3340 ms | 7.5418 ms | **1.24x** |
| `ac_rope_attention_train` | 14.1698 ms | 11.6688 ms | **1.21x** |

This is the strongest kernel story in the final codebase because it is no longer just a microbenchmark story. The final version improves real training-path execution.

## 2. Batched Action-Conditioned Attention

The action-conditioned attention path had a structural problem in the baseline: it processed action tokens one by one.

That means repeated:

- QKV projection setup,
- RoPE setup,
- small kernel launches,
- Python loop overhead.

The final code keeps the same logical token order but batches the action-token work into one larger path. The result is one of the clearest block-level wins in the codebase.

| Benchmark | Baseline | Final | Speedup |
|:---|---:|---:|---:|
| `ac_rope_attention_forward` | 10.0614 ms | 6.2051 ms | **1.62x** |

This is a good example of what improved the baseline most often: not replacing the whole operator, just removing repeated work from a real caller path.

## 3. Predictor And Masking Path Improvements

The next durable wins came from tensor plumbing.

### Predictor Broadcast And Gather Cleanup

The predictor path had repeated copies and reorder work that did not need to be written that way. The final code uses simpler broadcast and gather logic instead of materializing more tensors than necessary.

| Benchmark | Baseline | Final | Speedup |
|:---|---:|---:|---:|
| `src predictor forward` | 3.5376 ms | 2.7554 ms | **1.28x** |

### `apply_masks`

`apply_masks` is one of the cleaner retained wins. The final code recognizes the common case where all masks have the same shape and uses one batched gather instead of repeating gather work per mask group.

| Benchmark | Baseline | Final | Speedup |
|:---|---:|---:|---:|
| same-shape 1D masks | 0.2736 ms | 0.1323 ms | **2.07x** |
| same-shape 2D masks | 0.2104 ms | 0.1096 ms | **1.92x** |

### `repeat_interleave_batch`

The original implementation used nested Python concatenation. The final version expresses the same operation with shape transforms:

- reshape,
- expand,
- reshape back.

| Benchmark | Baseline | Final | Speedup |
|:---|---:|---:|---:|
| `repeat_interleave_batch` | 186.69 ms | 96.76 ms | **1.93x** |

These are not flashy changes, but they are the kind of improvements that make the baseline cleaner and faster at the same time.

## 4. Deterministic Helpers And Small Forward Fixes

The simplest final win is still worth showing because it is so large.

### Cached Action Block Causal Mask

The action block causal mask depends only on sequence structure. Rebuilding it every call was wasted work. The final baseline caches the mask and clones it on use.

| Benchmark | Baseline | Final | Speedup |
|:---|---:|---:|---:|
| action block causal mask build | 24.80 ms | 0.78 ms | **31.79x** |

### `AttentivePooler`

The final baseline uses `expand` instead of `repeat` for shared query tokens:

| Benchmark | Baseline | Final | Speedup |
|:---|---:|---:|---:|
| `attentive_pooler forward` | 2.2042 ms | 1.9425 ms | **1.14x** |

### Encoder Single-Mask Path

The final encoder path avoids concatenation when there is already exactly one mask:

| Benchmark | Baseline | Final | Speedup |
|:---|---:|---:|---:|
| `masked vit_tiny forward` | 10.5897 ms | 10.4081 ms | **1.017x** |

These are smaller than the RoPE and masking-path changes, but they are simple, correct, and cheap to keep.

## Why The End-To-End Gain Is Small

The short full-model check is small because the expensive part of V-JEPA 2 was never the easiest part to improve. The dense matrix work inside attention and MLP blocks was already fairly efficient. The retained improvements mostly cut:

- setup work,
- indexing work,
- tensor assembly,
- repeated helper work around the main blocks.

That produces real wins in local paths and sometimes clear wins in caller blocks, but the whole model is still dominated by the large dense kernels that were already good.

That is why the final picture looks like this:

- several strong wins between `1.2x` and `2.0x` on real subpaths,
- one very large deterministic-helper win,
- a strong retained Triton training-path improvement,
- but only a small overall movement on the short end-to-end check.

## Final Takeaway

The final V-JEPA 2 baseline got better in the places where it was actually wasting work:

- RoPE rotation that now survives the training path,
- action-conditioned attention that is batched instead of looped,
- predictor and masking code that moves less data around,
- deterministic helpers that are no longer recomputed every call.

The result is a baseline that is cleaner, more efficient, and easier to trust. The end-to-end gain is modest, but the retained code changes are real improvements to the baseline, and the strongest of them are exactly the kinds of changes that good engineers usually want to keep: faster code with simpler behavior and no loss of correctness.
