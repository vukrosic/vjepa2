#!/usr/bin/env python3
"""
V-JEPA 2 Kernel Queue Runner

Processes kernel experiments from queue/pending.jsonl:
1. Runs parity tests (pytest)
2. If parity passes, runs benchmark
3. Saves results to queue/results/{id}.json
4. Moves entry to queue/completed.jsonl

Usage:
    python queue_runner.py              # process all pending
    python queue_runner.py --watch      # watch mode: poll every 10s for new entries
    python queue_runner.py --id NAME    # run only a specific entry
"""

import argparse
from collections import Counter
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent
QUEUE_DIR = ROOT / "queue"
PENDING = QUEUE_DIR / "pending.jsonl"
COMPLETED = QUEUE_DIR / "completed.jsonl"
RESULTS_DIR = QUEUE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _compact_output(output, limit=240):
    text = " ".join(str(output or "").split())
    return text[:limit]


def classify_result(result):
    status = result.get("status")
    text = result.get("parity_output") or result.get("benchmark_output") or ""
    excerpt = _compact_output(text)

    if status == "APPROVED":
        return None, ""
    if status == "REJECTED":
        return "BENCH_REGRESSION", excerpt
    if status == "PARTIAL_WIN":
        return "PARTIAL_BENCH_WIN", excerpt

    lower = text.lower()
    if "not found" in lower:
        return "MISSING_ARTIFACT", excerpt
    if "syntaxerror" in lower or "indentationerror" in lower or "error collecting" in lower or "importtestmodule" in lower:
        return "IMPORT_OR_SYNTAX_ERROR", excerpt
    if "triton.compiler.errors" in lower or "compilationerror" in lower:
        return "TRITON_COMPILE_ERROR", excerpt
    if "mask argument cannot be block type" in lower or "unsupported ptr type" in lower:
        return "TRITON_POINTER_ERROR", excerpt
    if "illegal memory access" in lower or "cuda error" in lower:
        return "CUDA_RUNTIME_ERROR", excerpt
    if "save_for_backward can only save variables" in lower or "autograd.function" in lower:
        return "AUTOGRAD_ERROR", excerpt
    if "baseline_fn" in text or "keyerror:" in lower:
        return "BASELINE_OR_TEST_BUG", excerpt
    if "tensor-likes are not close" in lower:
        return "NUMERICAL_MISMATCH", excerpt
    if status == "FAILED_BENCHMARK":
        return "BENCHMARK_ERROR", excerpt
    if status == "FAILED_PARITY":
        return "PARITY_ERROR", excerpt
    return "UNKNOWN", excerpt


def finalize_result(result):
    category, excerpt = classify_result(result)
    result["failure_category"] = category
    result["failure_excerpt"] = excerpt
    return result


def load_pending():
    if not PENDING.exists():
        return []
    entries = []
    with open(PENDING) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Normalize: ensure id, kernel, test, bench fields exist
            # Handle "name" field (from some agents) and "kernel" field (full path)
            kernel_raw = e.get("kernel") or e.get("name") or "unknown"
            # Extract just the kernel name from path like "src/models/utils/kernels/fused_abs.py"
            kernel_name = pathlib.Path(kernel_raw).stem.replace("fused_", "fused_")
            if not kernel_name.startswith("fused_"):
                kernel_name = f"fused_{kernel_name}"
            e["kernel"] = kernel_name
            if "id" not in e:
                # Count existing entries for this kernel
                existing = sum(1 for x in entries if x.get("kernel") == kernel_name)
                e["id"] = f"{kernel_name}_{existing + 1:03d}"
            e.setdefault("test", f"tests/queue/test_{e['kernel']}.py")
            e.setdefault("bench", f"benchmarks/queue/bench_{e['kernel']}.py")
            e.setdefault("note", "")
            entries.append(e)
    return entries


def save_pending(entries):
    with open(PENDING, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def append_completed(entry, result):
    entry["result"] = result
    entry["completed_at"] = datetime.now().isoformat()
    with open(COMPLETED, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_parity_test(test_file):
    """Run pytest on the parity test. Returns (passed: bool, output: str)."""
    test_path = ROOT / test_file
    if not test_path.exists():
        return False, f"Test file not found: {test_file}"

    # Clear Triton JIT cache so latest source changes are compiled
    subprocess.run(["rm", "-rf", str(ROOT / ".triton")], capture_output=True)
    triton_cache = pathlib.Path.home() / ".triton" / "cache"
    subprocess.run(["rm", "-rf", str(triton_cache)], capture_output=True)

    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = ""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-x", "-q", "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        timeout=120,
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    return passed, output


def run_benchmark(bench_file):
    """Run benchmark script. Returns (results: dict|None, output: str)."""
    bench_path = ROOT / bench_file
    if not bench_path.exists():
        return None, f"Benchmark file not found: {bench_file}"

    result = subprocess.run(
        [sys.executable, str(bench_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=300,
    )
    output = result.stdout + result.stderr

    # Extract BENCH_RESULT JSON from output
    bench_data = None
    for line in result.stdout.splitlines():
        if line.startswith("BENCH_RESULT="):
            try:
                bench_data = json.loads(line[len("BENCH_RESULT="):])
            except json.JSONDecodeError:
                pass

    return bench_data, output


def process_entry(entry):
    entry_id = entry["id"]
    print(f"\n{'='*60}")
    print(f"Processing: {entry_id}")
    print(f"  Kernel: {entry.get('kernel', '?')}")
    print(f"  Description: {entry.get('description', '?')}")
    print(f"  Target: {entry.get('target_file', '?')}:{entry.get('target_lines', '?')}")
    print(f"{'='*60}")

    result = {
        "id": entry_id,
        "status": "unknown",
        "parity_passed": False,
        "parity_output": "",
        "benchmark": None,
        "benchmark_output": "",
        "timestamp": datetime.now().isoformat(),
    }

    # Step 1: Parity test
    print(f"\n[1/2] Running parity test: {entry['test']}")
    passed, output = run_parity_test(entry["test"])
    result["parity_passed"] = passed
    result["parity_output"] = output

    if not passed:
        result["status"] = "FAILED_PARITY"
        finalize_result(result)
        print(f"  FAILED parity test")
        print(f"  Category: {result['failure_category']}")
        print(f"  Output: {output[:500]}")
        save_result(entry_id, result)
        return result

    print(f"  PASSED parity test")

    # Step 2: Benchmark
    print(f"\n[2/2] Running benchmark: {entry['bench']}")
    bench_data, output = run_benchmark(entry["bench"])
    result["benchmark"] = bench_data
    result["benchmark_output"] = output

    if bench_data is None:
        result["status"] = "FAILED_BENCHMARK"
        print(f"  FAILED benchmark (no results extracted)")
        print(f"  Output: {output[:500]}")
    else:
        # Check if any shape shows speedup > 1.0
        any_win = any(v.get("speedup", 0) > 1.02 for v in bench_data.values())
        all_win = all(v.get("speedup", 0) > 1.0 for v in bench_data.values())

        if all_win:
            result["status"] = "APPROVED"
        elif any_win:
            result["status"] = "PARTIAL_WIN"
        else:
            result["status"] = "REJECTED"

        print(f"\n  Results:")
        for shape, data in bench_data.items():
            status = "WIN" if data.get("speedup", 0) > 1.02 else "LOSS" if data.get("speedup", 0) < 0.98 else "NEUTRAL"
            print(f"    {shape}: {data['baseline_ms']:.4f} ms -> {data['kernel_ms']:.4f} ms ({data['speedup']:.2f}x) [{status}]")
        print(f"\n  Verdict: {result['status']}")

    finalize_result(result)
    if result["failure_category"]:
        print(f"  Category: {result['failure_category']}")
    save_result(entry_id, result)
    return result


def save_result(entry_id, result):
    result_path = RESULTS_DIR / f"{entry_id}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Result saved: {result_path}")


def print_summary(results):
    print(f"\n{'='*60}")
    print(f"QUEUE SUMMARY")
    print(f"{'='*60}")

    approved = [r for r in results if r["status"] == "APPROVED"]
    partial = [r for r in results if r["status"] == "PARTIAL_WIN"]
    rejected = [r for r in results if r["status"] == "REJECTED"]
    failed_parity = [r for r in results if r["status"] == "FAILED_PARITY"]
    failed_bench = [r for r in results if r["status"] == "FAILED_BENCHMARK"]

    print(f"  APPROVED:       {len(approved)}")
    print(f"  PARTIAL_WIN:    {len(partial)}")
    print(f"  REJECTED:       {len(rejected)}")
    print(f"  FAILED_PARITY:  {len(failed_parity)}")
    print(f"  FAILED_BENCH:   {len(failed_bench)}")
    print(f"  TOTAL:          {len(results)}")

    categories = Counter()
    for result in results:
        category = result.get("failure_category")
        if category is None and result.get("status") != "APPROVED":
            category, _ = classify_result(result)
        if category:
            categories[category] += 1

    if categories:
        print(f"\nFailure categories:")
        for category, count in categories.most_common():
            print(f"  {category}: {count}")

    if approved:
        print(f"\nApproved kernels:")
        for r in approved:
            best = max(r["benchmark"].values(), key=lambda v: v.get("speedup", 0))
            print(f"  {r['id']}: best {best['speedup']:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="V-JEPA 2 Kernel Queue Runner")
    parser.add_argument("--watch", action="store_true", help="Watch mode: poll for new entries")
    parser.add_argument("--id", type=str, help="Run only a specific entry by ID")
    parser.add_argument("--summary", action="store_true", help="Print summary of completed results")
    args = parser.parse_args()

    if args.summary:
        results = []
        for f in RESULTS_DIR.glob("*.json"):
            with open(f) as fh:
                results.append(json.load(fh))
        print_summary(results)
        return

    if args.watch:
        print("Watch mode: polling for new entries every 10s...")
        processed_ids = set()
        while True:
            entries = load_pending()
            new_entries = [e for e in entries if e.get("id") not in processed_ids]
            if new_entries:
                all_results = []
                remaining = []
                for entry in entries:
                    if entry.get("id") in processed_ids:
                        continue
                    result = process_entry(entry)
                    all_results.append(result)
                    processed_ids.add(entry.get("id"))
                    append_completed(entry, result)

                # Remove processed from pending
                remaining = [e for e in entries if e.get("id") not in processed_ids]
                save_pending(remaining)

                if all_results:
                    print_summary(all_results)
            time.sleep(10)
    else:
        entries = load_pending()
        if args.id:
            entries = [e for e in entries if e["id"] == args.id]
            if not entries:
                print(f"No pending entry with id: {args.id}")
                return

        if not entries:
            print("No pending entries in queue.")
            return

        print(f"Processing {len(entries)} pending entries...")
        all_results = []
        processed_ids = set()

        for entry in entries:
            result = process_entry(entry)
            all_results.append(result)
            processed_ids.add(entry["id"])
            append_completed(entry, result)

        # Remove processed from pending
        remaining = load_pending()
        remaining = [e for e in remaining if e["id"] not in processed_ids]
        save_pending(remaining)

        print_summary(all_results)


if __name__ == "__main__":
    main()
