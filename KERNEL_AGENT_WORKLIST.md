# Kernel Agent Worklist

This is the current allowlist for simpler agents.

If a family is not listed here, do not touch it.

## Locked / Done

These are already queued or intentionally retired.

Do not touch them unless a human explicitly asks for a rerun or cleanup.

- `fused_action_block_causal_attention_mask`
- `fused_apply_masks_source`
- `fused_max_tensor`
- `fused_min_tensor`
- `fused_qkv_split`
- `fused_repeat_interleave_batch`
- `fused_rotate_queries_or_keys`
- `fused_rotate_query_key_pair`
- `fused_sorted_target_positions`

Notes:

- `fused_max_tensor` and `fused_min_tensor` are trivial exact wrappers and are not active kernel work.
- `fused_qkv_split` has been refreshed against the current code and re-enqueued.

## Priority 1: Exact Fixes Only

Use these for narrow reruns and cleanup only:

- `fused_3d_sincos_embed`
  Goal: preserve exact predictor shape semantics and flattening order.
- `fused_rms_residual`
  Goal: preserve the reference-backed path and confirm honest benchmark behavior.

Expected approach:

- do not invent variants
- do not add complexity
- fix imports, parity, fallback behavior, or benchmark honesty only

## Priority 2: Rerun Or Contain

Use these only if Priority 1 is already handled or a human directs you there:

- `fused_online_softmax`
  Goal: rerun the current passing implementation and keep the benchmark story honest.

Expected approach:

- correctness first
- fallback is acceptable
- do not fake a win
- do not add speculative Triton tricks

## Explicitly Forbidden Right Now

Do not spend agent time on these without fresh human approval:

- `fused_add_norm_residual`
  Reason: current family is not an exact source-backed match and has regressed locally.
- `fused_gelu_linear`
  Reason: current family is not a real fused projection and drops training gradients.
- tiny standalone unary kernels
- tiny standalone binary kernels
- any new family not named above

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
