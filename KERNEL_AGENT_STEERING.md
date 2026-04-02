# Kernel Agent Steering

Paste this into another agent when you want one narrow, safe kernel task.

```text
You are working in /workspace/vjepa2.

Read these files first and follow them exactly:
- AGENTS.md
- KERNEL_AGENT_INSTRUCTIONS.md
- KERNEL_AGENT_WORKLIST.md
- KERNEL_PRIMOPS.md

Ignore every other markdown file unless a human points you at it.

Your job is to complete exactly one family from the current worklist.

Hard rules:
- Do not edit queue state by hand.
- Do not create a new kernel family.
- Do not touch any family marked `Locked / Done` in `KERNEL_AGENT_WORKLIST.md`.
- Do not guess shapes, ranks, dtypes, or backward requirements.
- Keep baseline_fn exact.
- Keep kernel_fn guarded and safe.
- Do not use --skip-parity unless a human explicitly tells you to.
- If parity is not proven, stop before enqueueing.

Environment note:
- `pytest` works here.
- `timm` works here.
- You are expected to run real parity locally.

Required workflow:
1. Read the source op, kernel, test, benchmark, and recent same-family results.
2. State the exact shape, dtype, layout, and backward assumptions before changing code.
3. Implement the smallest safe fix.
4. Run `python scripts/prequeue_validate.py --kernel KERNELNAME --run-parity`.
5. If parity passes and benchmark matters, run `python scripts/prequeue_validate.py --kernel KERNELNAME --run-parity --run-bench`.
6. Enqueue only with `python scripts/enqueue_kernel.py ...`.

If pytest is unavailable in this environment, do not fake parity. Leave the change ready for a real parity run and report that clearly.

Your output should be:
- the touched files
- the validation command(s) you ran
- whether the family is ready to enqueue
```
