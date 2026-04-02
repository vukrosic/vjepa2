# Kernel Primops

This file is the allowlist for simpler agents.

If your kernel pattern is not clearly one of these, do not write it.

## Allowed Patterns

### 1. Exact Reference-Backed Helper

Use when:
- the existing Triton path is broken
- the baseline is easy to express in PyTorch
- correctness matters more than speed right now

Rules:
- keep `baseline_fn()` exact
- keep `kernel_fn()` as a guarded wrapper
- prefer safe fallback over fragile custom autograd

### 2. Row-Wise Reduction

Use for:
- softmax
- row mean / sum
- RMS or variance style reductions

Rules:
- reduce across one obvious contiguous axis
- accumulate in fp32 when numerics matter
- initialize accumulators before loops

### 3. Norm + Residual

Use for:
- LayerNorm + residual
- RMSNorm + residual

Rules:
- explicit shape checks
- fp32 stats
- exact backward or fallback

### 4. Attention Layout Helper

Use for:
- QKV split
- reshape / transpose helpers
- RoPE on already well-defined contiguous layouts

Rules:
- make head, sequence, and feature axes explicit
- keep indexing simple
- prefer separate helpers over giant fused attention kernels

### 5. Gather / Scatter Helper

Use for:
- token gather
- token scatter
- index-select style helpers

Rules:
- verify index dtype and bounds assumptions
- keep pointer arithmetic obvious
- stop if the indexing logic becomes tricky

## Current Bias

For this repo, the safest first move is usually:

1. exact reference-backed helper
2. row-wise reduction
3. norm + residual
4. attention layout helper

If you are about to write a tiny standalone unary or binary Triton kernel, stop. That is almost always the wrong job right now.

## Banned Patterns

Do not write these without explicit human approval:
- tiny standalone unary kernels
- tiny standalone binary kernels
- speculative mega-fusions
- RNG-heavy kernels
- complicated custom backward kernels
- kernels that mix gather, reduction, and layout transforms in one first pass

## Triton Safety Rules

- `tl.arange` only with `tl.constexpr` sizes
- block masks only with block pointers
- initialize every accumulator
- keep pointer math linear and easy to audit
- if support is uncertain, fall back to baseline
