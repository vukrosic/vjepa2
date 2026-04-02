---
title: "Optimizing Meta's V-JEPA 2: 31x Wins, Triton Kernels, and 12 Rejected Ideas"
date: "2026-04-02"
description: "I optimized Meta's V-JEPA 2 video foundation model with Triton kernels, PyTorch vectorization, and aggressive caching — achieving up to 31x on mask construction, 6.24x on RoPE rotation, and 1.24x on training attention blocks. More than half the ideas were rejected after honest benchmarking."
readTime: "12 min read"
tags: ["V-JEPA 2", "Triton", "Kernel Optimization", "Video Models", "RTX 3090", "Meta"]
sortOrder: 0
---

I optimized [Meta's V-JEPA 2](https://github.com/facebookresearch/vjepa2) — a video foundation model that learns visual representations through masked prediction. The [optimized fork](https://github.com/vukrosic/vjepa2) contains Triton kernels, PyTorch vectorization improvements, and caching strategies that together speed up core operations by up to **31x**. But the real story is about discipline: more than half the ideas I tried were rejected after honest benchmarking.

The rule I followed: an optimization only survives if it preserves exact behavior, passes parity tests, and wins on the actual hot path — not just in a microbenchmark.

## Where Time Was Actually Going

V-JEPA 2 uses a masked video prediction architecture with Rotary Position Embeddings (RoPE), action-conditioned attention, and multi-mask training. Profiling the forward pass on an RTX 3090 revealed that the heavy attention core was already fast — PyTorch's SDPA handles the GEMMs well. The opportunities were in the code *around* attention:

- Repeated Python-side mask construction on every call
- Redundant RoPE frequency and position computation
- Extra tensor copies from `repeat()` and `cat()` where broadcasting would work
- Action-conditioned attention glue around otherwise efficient kernels

This is a common pattern. The best optimization targets in mature codebases are rarely the core compute — they're the repeated setup work that nobody profiled.

## What Actually Worked

### Cached Action Causal Mask — 31.79x

The biggest single win was also the simplest. V-JEPA 2 builds a block-causal attention mask for action-conditioned training. The original implementation used nested Python loops to fill block regions on every forward call. The mask depends only on the sequence structure (`T`, `H`, `W`, `add_tokens`), which doesn't change during training.

The fix: build the mask once and cache it.

```python
@lru_cache(maxsize=None)
def _cached_action_block_causal_attention_mask(T, H, W, add_tokens):
    N_T = add_tokens + (H * W)
    frame_mask = torch.ones((T, T), dtype=torch.bool).tril()
    return frame_mask.repeat_interleave(N_T, dim=0).repeat_interleave(N_T, dim=1)

def build_action_block_causal_attention_mask(T, H, W, add_tokens=1):
    return _cached_action_block_causal_attention_mask(T, H, W, add_tokens).clone()
```

| Kernel | Baseline | Optimized | Speedup |
|:---|---:|---:|---:|
| Action causal mask build | 24.80 ms | 0.78 ms | **31.79x** |

The `.clone()` ensures each caller gets its own copy. This is the kind of optimization that looks obvious in retrospect but gets missed because it's "just mask construction."

### Faster Batch Repeat-Interleave — 1.90x

The `repeat_interleave_batch` utility was using nested `torch.cat()` in a Python loop. Replacing it with reshape-expand-reshape vectorization:

```python
def repeat_interleave_batch(x, B, repeat):
    if repeat == 1:
        return x
    N = len(x) // B
    return (x.reshape(N, B, *x.shape[1:])
             .unsqueeze(1)
             .expand(N, repeat, B, *x.shape[1:])
             .reshape(N * repeat * B, *x.shape[1:]))
```

| Kernel | Baseline | Optimized | Speedup |
|:---|---:|---:|---:|
| `repeat_interleave_batch` | 189.03 ms | 99.48 ms | **1.90x** |

No custom kernel. Just letting PyTorch avoid materializing intermediate tensors.

### Batched Multi-Mask Gathering — 1.92x / 2.07x

V-JEPA 2's `apply_masks` function selects token subsets using index masks. The original ran one `torch.gather` per mask group. When all masks have the same shape — which they do in the common training case — you can stack them and run a single batched gather.

| Benchmark | Baseline | Optimized | Speedup |
|:---|---:|---:|---:|
| `apply_masks` (4× 2D masks) | 0.2104 ms | 0.1096 ms | **1.92x** |
| `apply_masks` (4× 1D masks) | 0.2736 ms | 0.1323 ms | **2.07x** |

Again, no Triton. The fast path triggers only when conditions are met (multiple masks, same shape, same feature dim) and falls back to the original per-mask gather otherwise.

### RoPE Caching and Math Cleanup

Several incremental RoPE improvements survived together:

- **Inverse frequency caching**: the `1/10000^(2i/d)` computation is deterministic per `(device, dtype, dim)` tuple. Cache it in a module-level dict instead of recomputing every block.
- **Position caching**: separated `d/h/w` position tensors cached similarly.
- **Einsum removal**: replaced `torch.einsum` with direct broadcasted multiply.
- **Fused q/k rotation**: rotate query and key together through a shared helper instead of two separate calls.

These are individually small. Honest assessment: they help RoPE blocks, the action-conditioned path benefits more than the plain encoder, and some are close to neutral at short full-model scale. Not a license to overclaim — but useful.

## Triton Kernels

### Fused Query-Key RoPE Pair Kernel — 6.24x Primitive

The clear Triton win was fusing query and key rotation into a single kernel. The old code concatenated q and k, rotated the packed tensor, then split it back. The Triton kernel does both rotations in one pass — one position load, one sin/cos evaluation, two outputs:

```python
@triton.jit
def rope_rotate_pair_kernel(
    Q, K, Out_Q, Out_K, Pos, Omega,
    stride_q_n, stride_q_d,
    stride_k_n, stride_k_d,
    stride_oq_n, stride_oq_d,
    stride_ok_n, stride_ok_d,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D

    # Load position and compute sin/cos once
    pos = tl.load(Pos + row).to(tl.float32)
    omega = tl.load(Omega + cols // 2, mask=mask)
    angle = pos * omega
    cos_val = tl.cos(angle)
    sin_val = tl.sin(angle)

    # Pair trick: swap adjacent elements
    pair_cols = cols ^ 1
    sign = tl.where(cols % 2 == 0, -1.0, 1.0)

    # Rotate Q
    q = tl.load(Q + row * stride_q_n + cols * stride_q_d, mask=mask)
    q_pair = tl.load(Q + row * stride_q_n + pair_cols * stride_q_d, mask=mask)
    out_q = q * cos_val + q_pair * sign * sin_val
    tl.store(Out_Q + row * stride_oq_n + cols * stride_oq_d, out_q, mask=mask)

    # Rotate K (reuse same sin/cos)
    k = tl.load(K + row * stride_k_n + cols * stride_k_d, mask=mask)
    k_pair = tl.load(K + row * stride_k_n + pair_cols * stride_k_d, mask=mask)
    out_k = k * cos_val + k_pair * sign * sin_val
    tl.store(Out_K + row * stride_ok_n + cols * stride_ok_d, out_k, mask=mask)
```

The `cols ^ 1` trick handles the pair indexing without branches — for even indices it grabs the odd neighbor, and vice versa.

| Level | Baseline | Optimized | Speedup |
|:---|---:|---:|---:|
| Primitive (`rotate_query_key_pair`) | 1.3617 ms | 0.2182 ms | **6.24x** |
| Caller (`ac_rope_attention_forward`) | 2.0785 ms | 1.9283 ms | **1.08x** |
| Caller (`rope_attention_forward`) | 1.1694 ms | 1.2025 ms | 0.97x |

This illustrates the three-level rule perfectly. The primitive wins big. At the caller level, the action-conditioned path improves. The plain RoPE path doesn't — so the kernel is retained for the AC path but doesn't earn a claim on the plain encoder.

### Launch Config Tuning — Up to 50% Free Performance

After the Triton RoPE kernel was working and passing parity, a launch parameter sweep found more headroom without changing the math:

- `rotate_queries_or_keys`: `num_warps=2` (down from default 4)
- `rotate_query_key_pair`: `block_d=128, num_warps=2`

| Benchmark | Old Config | Tuned Config | Improvement |
|:---|---:|---:|---:|
| `rotate_queries_or_keys` | 0.9450 ms | 0.4668 ms | **50.57%** |
| `rotate_query_key_pair` | 1.9178 ms | 1.3901 ms | **27.53%** |

The lesson: write the math first, sweep launch configs before rewriting kernels. The same code can perform dramatically differently with different warp and block size settings.

### Training-Safe Autograd Wrapper — 1.24x on Training

An inference-only Triton kernel isn't enough for a training codebase. RoPE is an orthonormal rotation, so the gradient is the same rotation applied with the negated position angle. That means the backward pass can reuse the exact same Triton kernel:

```python
class _TritonRotateQueryKeyPairAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, pos, omega):
        ctx.save_for_backward(pos, omega)
        return triton_rotate_query_key_pair(q, k, pos, omega)

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        pos, omega = ctx.saved_tensors
        # Inverse rotation = same kernel with -pos
        grad_q, grad_k = triton_rotate_query_key_pair(
            grad_q.contiguous(), grad_k.contiguous(), -pos, omega
        )
        return grad_q, grad_k, None, None
```

This is the conceptual insight that made the Triton path viable for real training. The first backward attempt failed because the repo uses a compatibility-preserving RoPE layout where the naive `-pos` inverse wasn't mathematically valid. After understanding the actual rotation structure, the correct backward worked.

| Benchmark | Baseline | Optimized | Speedup |
|:---|---:|---:|---:|
| `rope_attention_train` (F+B) | 9.3340 ms | 7.5418 ms | **1.24x** |
| `ac_rope_attention_train` (F+B) | 14.1698 ms | 11.6688 ms | **1.21x** |
| `rotate_queries_or_keys_train` (F+B) | 2.1162 ms | 0.9759 ms | **2.17x** |

The higher bar for manual kernels: exact backward parity, verified with gradient checks, winning at the caller level — not just the primitive.

## Correctness Fixes (Not Speed, But Essential)

Two changes were about correctness, not performance:

1. **SDPA dropout in eval mode**: the attention modules were passing a nonzero dropout probability into `F.scaled_dot_product_attention` during `eval()`. This made eval output non-deterministic. Fixed by gating dropout on `self.training`.

2. **Half-precision RoPE**: cached position tensors were created in `float32` but fed into `float16` attention paths. This caused silent precision degradation. Fixed by matching the position tensor dtype to the input.

These matter because optimization claims are worthless if the baseline path is buggy.

## What Was Rejected (And Why)

This is the section I think matters most. More than half the ideas I tried didn't survive the keep/reject filter. Each one sounded reasonable, some showed impressive microbenchmark numbers, and all were rejected for the same reason: they didn't win on the actual hot path.

### Triton `apply_masks` Gather Kernel

A fused Triton gather kernel for mask application looked promising:

| Level | Baseline | Optimized | Speedup |
|:---|---:|---:|---:|
| Primitive | 0.2958 ms | 0.2611 ms | 1.13x |
| Masked image encoder | 52.0714 ms | 52.1849 ms | 0.998x |
| Masked video encoder | 40.1243 ms | 40.2205 ms | 0.998x |

The primitive won. The callers didn't. Rejected.

### Fused Multi-Axis RoPE Triton Kernel

A kernel that fuses the depth/height/width rotations into one pass. Failed parity on realistic shapes before it ever earned a benchmark. Rejected.

### Contiguous Patch-Embed Output

Inserting a `.contiguous()` after patch embedding to help downstream operations. Mixed results:

| Benchmark | Baseline | Contiguous | Speedup |
|:---|---:|---:|---:|
| Image unmasked | 3.8271 ms | 3.7288 ms | 1.03x |
| Image masked | 3.7480 ms | 3.7693 ms | 0.99x |
| Video masked | 3.6584 ms | 3.7410 ms | 0.98x |

The main masked path regressed. Rejected.

### Predictor RoPE Position Precompute

Precomputing separated RoPE positions once per predictor forward instead of per-block:

| Benchmark | Dynamic Path | Precomputed | Speedup |
|:---|---:|---:|---:|
| `predictor_rope_internal` | 13.9165 ms | 14.2865 ms | 0.973x |

Parity passed. Benchmark said no. Rejected.

### Predictor Inverse-Permutation via Scatter

Replacing the second `argsort` with `scatter_` for inverse permutation:

| Benchmark | Baseline | Scatter | Speedup |
|:---|---:|---:|---:|
| B=8, N=512 | 0.0953 ms | 0.1078 ms | 0.884x |

`argsort` is faster at typical sizes. Rejected.

### Other Rejections

- **App predictor sorted-mask fast path**: 1.03x in one run, 1.00x averaged over four. Noise.
- **Direct predictor mask-token expansion**: 0.98x. Lost.
- **First Triton RoPE backward**: failed gradient parity completely. Wrong math for the layout.
- **Batched-position Triton extension**: parity failed on masked shapes. Can't tolerate numeric drift in training.
- **Separable RoPE Triton kernel**: needed fixes to compile, then failed parity anyway.
- **GQA / sparse attention / temporal aggregation kernels**: not on the live execution path, never earned benchmark time.

## The Three-Level Test

The pattern that emerged is what I call the three-level test:

1. **Primitive benchmark** — Does the isolated operation get faster? This is necessary but not sufficient.
2. **Block-level benchmark** — Does the containing module (attention block, predictor) actually speed up? Many primitive wins vanish here because the operation wasn't the bottleneck.
3. **Model-level check** — Does the default training/eval path move? Some block wins disappear when combined with other modules.

If an optimization only wins at Level 1, it's dead code. The Triton `apply_masks` kernel is the clearest example: 1.13x at the primitive, 0.998x at the caller. Not a win.

## Benchmarking Methodology

All measurements were taken on an RTX 3090 (sm_86) with:

- CUDA events for timing (not `time.perf_counter()`)
- 20-40 warmup iterations before measurement
- 30-200 timed iterations per benchmark
- `fp16` data types on realistic shapes (batch 2-8, seq 256-4096, heads 8-16)
- Comparison against `git show HEAD:path/to/file.py` for faithful baselines

Each retained optimization has a corresponding parity test that verifies exact numerical equivalence (within dtype tolerance: `5e-3` for fp16, `1e-5` for fp32). The test suite covers primitives, attention blocks, predictors, and gradient correctness.

## Lessons

1. **Caching beats kernels.** The biggest win (31x) came from `@lru_cache`, not Triton. Always look for repeated deterministic computation before writing custom kernels.

2. **Vectorization before custom code.** The `apply_masks` and `repeat_interleave` wins used pure PyTorch reshape/expand tricks. No custom kernels, no compilation, no risk.

3. **Launch config tuning is free performance.** Same Triton math with different `num_warps` and `block_d` yielded 27-50% improvements. Profile before rewriting.

4. **Backward correctness is the hard part.** The Triton RoPE forward was straightforward. Getting the backward right required understanding the specific RoPE layout — the first attempt failed completely.

5. **Primitive wins lie.** A 6.24x kernel speedup became 1.08x at the caller level. A 1.13x kernel win became 0.998x. Always measure at the level that matters.

6. **Document rejections.** I spent more time on ideas that didn't work than on ideas that did. Writing them down prevents rediscovering the same dead ends.

7. **The code around attention is where the opportunities are.** SDPA handles the core GEMMs. The gains are in mask construction, position encoding, tensor assembly, and the glue between modules.

## Summary of Retained Wins

| Optimization | Speedup | Type |
|:---|---:|:---|
| Cached action causal mask | **31.79x** | Python caching |
| Fused q/k RoPE pair kernel (primitive) | **6.24x** | Triton kernel |
| Batched 1D mask gathering | **2.07x** | PyTorch vectorization |
| Triton RoPE train (primitive, F+B) | **2.17x** | Triton + autograd |
| Batched 2D mask gathering | **1.92x** | PyTorch vectorization |
| Batch repeat-interleave | **1.90x** | PyTorch vectorization |
| RoPE attention training (F+B) | **1.24x** | Triton + autograd |
| AC RoPE attention training (F+B) | **1.21x** | Triton + autograd |
| AC RoPE attention inference | **1.08x** | Triton kernel |

The code is at [github.com/vukrosic/vjepa2](https://github.com/vukrosic/vjepa2). The optimization article and engineering diary are in the repo under `docs/optimization/OPTIMIZATION_ARTICLE.md` and `docs/optimization/OPTIMIZATION_NOTES.md`.
