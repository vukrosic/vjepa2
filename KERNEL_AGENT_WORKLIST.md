# Kernel Agent Worklist

This is the current allowlist for simpler agents.

If a family is not listed here, do not touch it.

## Priority 1: Finalize Repaired Families

Use these for narrow reruns and cleanup only:

- `fused_max_tensor`
  Goal: clear the old import-error history with the smallest possible helper.
- `fused_min_tensor`
  Goal: clear the old import-error history with the smallest possible helper.
- `fused_rms_residual`
  Goal: preserve the reference-backed path and confirm honest benchmark behavior.
- `fused_3d_sincos_embed`
  Goal: preserve exact shape semantics after the baseline fix.

Expected approach:

- do not invent variants
- do not add complexity
- fix imports, parity, fallback behavior, or benchmark honesty only

## Priority 2: Correctness-First Families

Use these only if Priority 1 is already handled or a human directs you there:

- `fused_online_softmax`
- `fused_add_norm_residual`
- `fused_qkv_split`
- `fused_gelu_linear`

Expected approach:

- correctness first
- fallback is acceptable
- do not fake a win
- do not add speculative Triton tricks

## Forbidden Work

Do not do any of these without explicit human approval:

- create a new kernel family
- touch standalone tiny unary kernels
- touch standalone tiny binary kernels
- copy from old brainstorm docs
- bulk-generate variants
- submit anything whose only story is "maybe Triton is faster"

## Exact Deliverable

For one family, produce:

- one corrected kernel file
- one matching queue test
- one matching benchmark
- the validation command(s) you ran
- a clear note saying whether it is ready to enqueue

If you finish one family cleanly, stop and report before starting another.
