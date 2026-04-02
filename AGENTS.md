# Agents

If you are working on kernel queue experiments, start here.

Read only these files first:

1. `AGENTS.md`
2. `KERNEL_AGENT_INSTRUCTIONS.md`
3. `KERNEL_AGENT_WORKLIST.md`
4. `KERNEL_PRIMOPS.md`
5. `KERNEL_AGENT_STEERING.md` only if you need a pasteable prompt

Ignore the historical queue report and all other markdown files unless a human explicitly points you at them.

Rules:

- work only on families listed in `KERNEL_AGENT_WORKLIST.md`
- treat `Locked / Done` families in that file as off-limits unless a human explicitly asks for a rerun
- do not edit queue JSON files by hand
- do not invent new families without explicit approval
- prefer exact guarded reference paths over fragile Triton code
- stop before enqueueing if parity is not actually proven
- this environment has working `pytest` and `timm`; run real parity locally
