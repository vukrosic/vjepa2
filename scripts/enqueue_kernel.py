#!/usr/bin/env python3
"""Validate and enqueue a kernel experiment safely.

This script exists to keep bad experiments out of `queue/pending.jsonl`.
It runs prequeue validation first, then appends a normalized queue entry
with an auto-generated id if validation succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
QUEUE_DIR = ROOT / "queue"
PENDING = QUEUE_DIR / "pending.jsonl"
COMPLETED = QUEUE_DIR / "completed.jsonl"
RESULTS_DIR = QUEUE_DIR / "results"
VALIDATOR = ROOT / "scripts" / "prequeue_validate.py"

ID_RE = re.compile(r"^(?P<kernel>.+)_(?P<num>\d{3})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and append a kernel entry to queue/pending.jsonl")
    parser.add_argument("--kernel", required=True, help="Kernel name, e.g. fused_gelu_linear")
    parser.add_argument("--description", required=True, help="Exact description of the experiment")
    parser.add_argument("--target-file", required=True, help="Source file this optimization targets")
    parser.add_argument("--target-lines", default="", help="Relevant source line span")
    parser.add_argument("--test", help="Override test path")
    parser.add_argument("--bench", help="Override benchmark path")
    parser.add_argument("--note", default="", help="Optional note for the queue entry")
    parser.add_argument("--run-bench", action="store_true", help="Run benchmark validation before enqueueing")
    parser.add_argument("--skip-parity", action="store_true", help="Skip parity validation")
    parser.add_argument("--dry-run", action="store_true", help="Print the entry without appending it")
    return parser.parse_args()


def run_validation(args: argparse.Namespace) -> None:
    cmd = [sys.executable, str(VALIDATOR), "--kernel", args.kernel]
    if args.test:
        cmd.extend(["--test-path", args.test])
    if args.bench:
        cmd.extend(["--bench-path", args.bench])
    if not args.skip_parity:
        cmd.append("--run-parity")
    if args.run_bench:
        cmd.append("--run-bench")

    proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def iter_existing_ids(kernel: str):
    if PENDING.exists():
        with open(PENDING) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_id = entry.get("id", "")
                match = ID_RE.match(entry_id)
                if match and match.group("kernel") == kernel:
                    yield int(match.group("num"))

    if COMPLETED.exists():
        with open(COMPLETED) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_id = entry.get("id", "")
                match = ID_RE.match(entry_id)
                if match and match.group("kernel") == kernel:
                    yield int(match.group("num"))

    if RESULTS_DIR.exists():
        prefix = f"{kernel}_"
        for path in RESULTS_DIR.glob(f"{prefix}*.json"):
            match = ID_RE.match(path.stem)
            if match and match.group("kernel") == kernel:
                yield int(match.group("num"))


def next_id(kernel: str) -> str:
    current = max(iter_existing_ids(kernel), default=0)
    return f"{kernel}_{current + 1:03d}"


def append_entry(entry: dict) -> dict:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    PENDING.touch(exist_ok=True)
    with open(PENDING, "a") as f:
        try:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            # Recompute under lock to reduce id collisions from concurrent submitters.
            entry["id"] = next_id(entry["kernel"])
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    return entry


def main() -> int:
    args = parse_args()
    target_path = pathlib.Path(args.target_file)
    if not target_path.is_absolute():
        target_path = (ROOT / target_path).resolve()
    if not target_path.exists():
        print(f"error: target file not found: {target_path}", file=sys.stderr)
        return 2

    run_validation(args)

    entry = {
        "id": next_id(args.kernel),
        "kernel": args.kernel,
        "test": args.test or f"tests/queue/test_{args.kernel}.py",
        "bench": args.bench or f"benchmarks/queue/bench_{args.kernel}.py",
        "target_file": args.target_file,
        "target_lines": args.target_lines,
        "description": args.description,
        "note": args.note,
    }

    if args.dry_run:
        print(json.dumps(entry, indent=2))
        print("\ndry run only; nothing appended")
        return 0

    entry = append_entry(entry)
    print(json.dumps(entry, indent=2))
    print(f"\nappended to {PENDING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
