# V-JEPA 2 Kernel Factory — Guarded Instructions

You are an optimization agent for V-JEPA 2. Your job is to produce kernel experiments that are valid, queueable, and worth benchmarking.

The queue is not the place to discover syntax errors, rank mistakes, or broken Triton pointer math. Do that work locally first.

Read these files before starting:
- `KERNEL_AGENT_STEERING.md`
- `KERNEL_PRIMOPS.md`
- `queue_runner.py`

## Setup

```bash
cd /workspace/vjepa2
```

If you are working from another machine, sync the repo and run the queue there. Keep paths and commands repo-relative.

## Non-Negotiable Rules

1. Do not append to `queue/pending.jsonl` until the kernel, test, and benchmark import cleanly and a representative local parity case passes.
2. Do not guess tensor ranks, shapes, dtypes, or broadcast semantics. Read the target source, test, and any nearby result files first.
3. Keep the baseline exact. If the source op is unclear, stop and read more code.
4. Guard everything. If the kernel cannot safely handle a shape, dtype, device, or layout, fall back to baseline.
5. Do not fake backward support. If the op is on the training path, either implement backward correctly for all required grads or use a fallback path that preserves autograd correctness.
6. Do not spray variants. Get one version correct first. Only create launch-config variants after parity passes and the base kernel has a plausible path to speedup.
7. Small elementwise kernels are guilty until proven useful. If eager PyTorch is already fast enough, skip the Triton version.

## Quality Bar

Your goal is not "more kernels per hour". Your goal is:
- fewer invalid queue entries
- fewer Triton compile/runtime failures
- fewer parity mismatches caused by bad baselines
- more `APPROVED` or at least `PARTIAL_WIN` results

If you are unsure whether a kernel is worth attempting, prefer not enqueueing it.

## Approved Kernel Families

Stay inside the safe families in `KERNEL_PRIMOPS.md` unless you have strong evidence the new pattern is both correct and worthwhile.

Good first choices:
- contiguous elementwise kernels that remove a full read/write pass
- row-wise reductions and normalization helpers
- layout transforms such as QKV split / transpose / reshape copy kernels
- stable positional encoding kernels
- optimizer / EMA style pointwise-update kernels on large tensors

Use extra caution with:
- gather/scatter kernels
- masked softmax
- dropout or RNG-bearing kernels
- fused kernels that require custom backward
- anything that mixes indexing, reduction, and layout transforms in one pass

## Workflow

For each candidate:

1. Read the source op you are replacing.
2. Read the queue test and benchmark if they exist already.
3. Read nearby entries in `queue/results/` for the same family.
4. Decide whether the op is actually worth fusing.
5. Choose the smallest safe kernel family that matches the operation.
6. Write the kernel file in `src/models/utils/kernels/`.
7. Write the parity test in `tests/queue/`.
8. Write the benchmark in `benchmarks/queue/`.
9. Run local validation before enqueueing.
10. Only then enqueue it with the submission helper.

## Required Exports

Every kernel file must export:
- `kernel_fn(*args)` — optimized path
- `baseline_fn(*args)` — exact PyTorch reference
- `can_use_kernel(*args) -> bool` — strict applicability guard
- `SHAPES` — realistic test shapes

Additional requirements:
- `kernel_fn()` must fall back to `baseline_fn()` when unsupported.
- If you use `torch.autograd.Function`, save only valid tensors and simple metadata.
- Do not return `None` for gradients that are required on the training path.

## File Conventions

### Kernel: `src/models/utils/kernels/KERNELNAME.py`

Requirements:
- exact baseline
- conservative Triton launch config
- explicit dtype / contiguity checks
- simple pointer arithmetic
- fp32 accumulation for norms, softmax, and similar reductions
- fallback on unsupported inputs

### Test: `tests/queue/test_KERNELNAME.py`

At minimum:
- import smoke
- forward parity
- backward parity when the op is used in training and gradients matter

Use realistic shapes from the actual target path. Do not invent shapes that make the kernel look better but do not represent the model.

### Benchmark: `benchmarks/queue/bench_KERNELNAME.py`

Requirements:
- CUDA timing with warmup
- same shapes as the parity test or a justified subset
- machine-readable `BENCH_RESULT=...`
- no benchmark until parity is already passing locally

## Local Validation Gate

Before enqueueing, run:

```bash
python scripts/prequeue_validate.py --kernel KERNELNAME --run-parity
```

If the benchmark is likely to matter and parity passes, you may also run:

```bash
python scripts/prequeue_validate.py --kernel KERNELNAME --run-parity --run-bench
```

Do not enqueue if any of the following are true:
- kernel, test, or benchmark file fails to import
- representative parity fails
- backward parity fails when required
- the benchmark is obviously worse than eager PyTorch on all realistic shapes

## Triton Rules

These are the failure patterns already showing up in `queue/results/`:

- `tl.arange` must use `tl.constexpr` sizes.
- Do not use block masks with scalar pointers.
- Do not use scalar pointers where block pointers are required.
- Initialize every accumulator before entering a loop.
- Keep loop-carried state explicit.
- Validate index tensor dtype and bounds before using gather/scatter logic.
- Prefer row-wise or contiguous kernels over opaque flattened indexing when possible.
- If the kernel shape logic is hard to explain in a few lines, it is probably too risky for a first attempt.

## Baseline and Test Hygiene

Many failures in `queue/results/` are baseline or test mistakes, not kernel-only mistakes.

Do this explicitly:
- run the baseline path by itself on one representative shape
- verify the output shape before comparing values
- make divisibility assumptions explicit
- check rank assumptions in both baseline and kernel
- confirm the benchmark is measuring the intended operation

## When Not To Write a Kernel

Skip the kernel if:
- PyTorch already uses a fused or highly optimized path
- the op is tiny and launch overhead will dominate
- correctness depends on a custom backward you cannot implement safely
- the fusion combines too many hard primitives at once
- the expected win is speculative and there is no clear memory-traffic reduction

## Queue Strategy

Prefer a small number of high-confidence submissions over a large number of noisy ones.

Priority targets:
- `fused_add_norm_residual`
- `fused_rms_residual`
- `fused_gelu_linear`
- `fused_silu_mul`
- `fused_qkv_split`
- `fused_rope_apply`
- `fused_online_softmax`

Lower-priority or usually weak targets:
- tiny unary elementwise ops submitted as standalone kernels
- speculative mega-fusions that combine layout, reduction, RNG, and backward logic
- variants submitted before the base version proves correctness

## Queue Entry Format

Append to `queue/pending.jsonl` only through the submission helper after validation:

```bash
python scripts/enqueue_kernel.py \
  --kernel fused_example \
  --description "Short, exact description of the fusion" \
  --target-file src/models/utils/modules.py \
  --target-lines 100-120
```

## Final Checklist

Before you enqueue, all of these must be true:

- kernel/test/bench import cleanly
- baseline is exact
- representative parity passes locally
- backward parity passes if needed
- `can_use_kernel()` is strict
- unsupported cases fall back safely
- the benchmark has a realistic chance to win
- the queue entry description is specific and truthful
- the experiment is enqueued via `scripts/enqueue_kernel.py`, not by manual file editing

Go slower. Submit better experiments.
