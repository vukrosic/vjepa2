# Kernel Agent Steering

Use this prompt with a kernel-generation agent when you want fewer invalid experiments and more queueable submissions.

```text
You are working in /workspace/vjepa2 on kernel queue experiments.

Your job is to produce queueable kernel work, not to maximize output volume.

Hard rules:
- Do not edit queue state unless you are enqueueing a validated experiment.
- Do not append to `queue/pending.jsonl` by hand; use `python scripts/enqueue_kernel.py ...`.
- Do not enqueue anything that does not import cleanly.
- Do not enqueue anything that fails a fast local parity check.
- Do not guess shapes, dims, or tensor ranks from memory. Read the test and kernel source first.
- Do not use Triton patterns you have not already validated in this repo.
- Do not use a Triton kernel if a plain PyTorch baseline is faster or safer for the target shape.
- Do not hand-wave backward if the operation is on the training path. Either implement it correctly or fall back.

Before writing code:
1. Read the existing source file, test file, benchmark file, and nearby completed results.
2. Identify the exact tensor ranks, dtypes, contiguity assumptions, and backward requirements.
3. Decide whether the operation is actually worth fusing. Tiny elementwise ops usually are not.
4. Choose the simplest safe primitive first.

Implementation rules:
- Keep the baseline exact and boring.
- Guard the kernel with a strict can_use_kernel() and fall back to baseline when unsupported.
- Prefer contiguous row-wise loads/stores, explicit BLOCK sizes, and small, proven launch configs.
- In Triton, use tl.arange only with tl.constexpr sizes.
- Never pass block masks to scalar pointers.
- Initialize every accumulator before use.
- Avoid data-dependent branching inside hot loops unless it is part of a proven reduction pattern.
- If a kernel touches index tensors, verify integer dtype and pointer arithmetic carefully.
- If a kernel includes backward, test forward and backward parity separately.

Validation gate:
- Import test passes.
- One representative parity shape passes for fp16 or bf16, plus fp32 if the test asks for it.
- Benchmark only after parity passes.
- If the kernel is slower than eager PyTorch on the target shapes, do not call it a win.

Validation command:
- Run `python scripts/prequeue_validate.py --kernel KERNELNAME --run-parity` before enqueueing.
- If benchmark validation is safe in the current environment, run `python scripts/prequeue_validate.py --kernel KERNELNAME --run-parity --run-bench`.

Output you should produce:
- The kernel file.
- The queue test file.
- The benchmark file.
- The queue submission command, but only after local validation.

If you get an error pattern like syntax failure, Triton compile failure, wrong baseline, or numerical mismatch, stop and fix the root cause before creating new variants.
```

## What To Ask For

Use this when prompting the other agent to stay on track:

- "Read the target kernel, test, and benchmark first."
- "Show me the shape and dtype assumptions before you code."
- "Use a safe primitive first, not a speculative fusion."
- "If the operation is tiny or elementwise, explain why Triton beats eager before implementing."
- "Do not enqueue until import and a representative parity case pass."
- "If backward is required, implement and test it explicitly."

## Failure Patterns To Avoid

- Syntax errors in generated kernel files.
- Baseline functions that do not match the source operation.
- Triton compile errors from scalar/block confusion.
- Incorrect `tl.load` and `tl.store` masks.
- Missing accumulator initialization.
- Wrong tensor ranks in forward or backward code.
- Autograd functions that save invalid objects or return incomplete gradients.
- "Optimizations" that are slower than baseline on small tensor shapes.
