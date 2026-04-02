# Kernel Agent Runbook

This is the canonical execution doc for kernel agents.

Read only these files before you work:

1. `AGENTS.md`
2. `KERNEL_AGENT_INSTRUCTIONS.md`
3. `KERNEL_AGENT_WORKLIST.md`
4. `KERNEL_PRIMOPS.md`

Ignore every other markdown file unless a human explicitly points you at it.

## Mission

Produce one small, correct, reviewable kernel change.

Do not optimize for volume.
Do not create speculative families.
Do not touch queue state casually.

## Hard Rules

1. Work only on families listed in `KERNEL_AGENT_WORKLIST.md`.
2. Treat `Locked / Done` families in that file as off-limits unless a human explicitly asks for a rerun.
3. Do not hand-edit `queue/pending.jsonl`, `queue/completed.jsonl`, or `queue/results/*.json`.
4. Do not invent a new kernel family unless a human explicitly asks.
5. Keep `baseline_fn()` exact and boring.
6. Keep `kernel_fn()` guarded and safe.
7. If backward matters, prove it or fall back.
8. One kernel family at a time. No variants until the base version is correct.
9. Do not use `--skip-parity` unless a human explicitly tells you to.

## Exact Workflow

1. Read the target source op, kernel, test, benchmark, and recent same-family result files.
2. Confirm shape, dtype, layout, and backward assumptions in plain language.
3. Implement the smallest safe fix.
4. Run local validation.
5. Enqueue only through `scripts/enqueue_kernel.py`, and only after validation passes.

## Validation

Default validation:

```bash
python scripts/prequeue_validate.py --kernel KERNELNAME --run-parity
```

If parity passes and the benchmark matters:

```bash
python scripts/prequeue_validate.py --kernel KERNELNAME --run-parity --run-bench
```

If this environment does not have `pytest`, that is not permission to fake parity.
In that case:

- do import smoke
- do direct local function smoke if possible
- leave the change ready for a real parity run
- do not enqueue unless a human explicitly approves the exception

Current environment note:

- `pytest` is installed
- `timm` is installed
- queue agents are expected to run real parity locally before enqueueing

## Status Names

Read queue status names literally:

- `PARITY_IMPORT_OR_SYNTAX_ERROR`: the files do not import cleanly
- `PARITY_BASELINE_OR_TEST_BUG`: the reference or test is wrong
- `PARITY_TRITON_COMPILE_ERROR`: the Triton kernel does not compile
- `PARITY_TRITON_POINTER_ERROR`: invalid pointer or mask usage
- `PARITY_NUMERICAL_MISMATCH`: outputs or gradients are wrong
- `BENCHMARK_REGRESSION`: parity passed but the kernel is slower
- `BENCHMARK_PARTIAL_WIN`: mixed result
- `BENCHMARK_WIN`: real win

## Definition Of Done

A kernel is ready only if all of these are true:

- kernel, test, and benchmark import cleanly
- baseline matches the source operation
- representative forward parity passes
- backward parity passes when required
- benchmark is honest
- unsupported inputs fall back safely

## Allowed File Touches

Normal kernel work should stay inside:
- `src/models/utils/kernels/`
- `tests/queue/`
- `benchmarks/queue/`

You may also use:
- `scripts/prequeue_validate.py`
- `scripts/enqueue_kernel.py`

Do not rewrite queue infrastructure or broad repo docs unless the task is specifically about workflow cleanup.

## Preferred Fix Style

Prefer this order:
1. Exact reference-backed helper
2. Safe guarded Triton kernel
3. Faster variant after parity is already proven

For broken or fragile kernels, a correct fallback-backed implementation is better than a broken Triton attempt.

## Stop List

Do not spend time on:
- tiny standalone unary or binary ops
- speculative mega-fusions
- kernels that already benchmark slower than eager on realistic shapes
- custom backward work you cannot prove correct
- stale import-error cleanup for trivial wrappers that are already fixed
- non-exact training-path rewrites such as the current `fused_gelu_linear` family

## Queue Submission

When a kernel is actually ready:

```bash
python scripts/enqueue_kernel.py \
  --kernel KERNELNAME \
  --description "Short exact description" \
  --target-file path/to/source.py \
  --target-lines START-END
```

If you are not sure whether a submission is ready, stop before enqueueing.
