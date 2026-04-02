#!/usr/bin/env python3
"""Triage `queue/results/*.json` and rank actionable failures.

The script is read-only with respect to queue state. It can optionally write
an overall Markdown summary for humans.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "queue" / "results"
DEFAULT_MARKDOWN = ROOT / "QUEUE_TRIAGE.md"
ID_SUFFIX_RE = re.compile(r"^(?P<family>.+)_(?P<num>\d{3})$")


ACTIONABLE_ORDER = {
    "IMPORT_OR_SYNTAX_ERROR": 0,
    "BASELINE_OR_TEST_BUG": 1,
    "TRITON_POINTER_ERROR": 2,
    "TRITON_COMPILE_ERROR": 3,
    "AUTOGRAD_ERROR": 4,
    "CUDA_RUNTIME_ERROR": 5,
    "NUMERICAL_MISMATCH": 6,
    "BENCHMARK_ERROR": 7,
    "BENCH_REGRESSION": 8,
    "PARTIAL_BENCH_WIN": 9,
    "PARITY_ERROR": 10,
    "UNKNOWN": 11,
}


@dataclass
class ResultRow:
    path: pathlib.Path
    data: dict
    status: str
    family: str
    category: str
    excerpt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize queue result failures by category and family.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory containing result JSON files")
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN), help="Write a Markdown summary to this path")
    parser.add_argument("--no-markdown", action="store_true", help="Do not write a Markdown file")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows to show per category")
    parser.add_argument("--families", type=int, default=15, help="Maximum kernel families to show per category")
    return parser.parse_args()


def compact(text: object, limit: int = 220) -> str:
    s = " ".join(str(text or "").split())
    return s[:limit]


def family_from_entry(entry: dict, path: pathlib.Path) -> str:
    kernel = entry.get("kernel") or entry.get("id") or path.stem
    match = ID_SUFFIX_RE.match(str(kernel))
    if match:
        return match.group("family")
    return str(kernel)


def classify(entry: dict) -> tuple[str, str]:
    status = str(entry.get("status", "<missing>"))
    text = entry.get("parity_output") or entry.get("benchmark_output") or ""
    lower = str(text).lower()

    if status == "APPROVED":
        return "APPROVED", ""
    if status == "REJECTED":
        return "BENCH_REGRESSION", compact(text)
    if status == "PARTIAL_WIN":
        return "PARTIAL_BENCH_WIN", compact(text)
    if status == "FAILED_BENCHMARK":
        return "BENCHMARK_ERROR", compact(text)

    if "not found" in lower:
        return "MISSING_ARTIFACT", compact(text)
    if "syntaxerror" in lower or "indentationerror" in lower or "error collecting" in lower or "importtestmodule" in lower:
        return "IMPORT_OR_SYNTAX_ERROR", compact(text)
    if "mask argument cannot be block type" in lower or "unsupported ptr type" in lower:
        return "TRITON_POINTER_ERROR", compact(text)
    if "triton.compiler.errors" in lower or "compilationerror" in lower:
        return "TRITON_COMPILE_ERROR", compact(text)
    if "save_for_backward can only save variables" in lower or "autograd.function" in lower:
        return "AUTOGRAD_ERROR", compact(text)
    if "illegal memory access" in lower or "cuda error" in lower:
        return "CUDA_RUNTIME_ERROR", compact(text)
    if "baseline_fn" in lower or "keyerror:" in lower or "valueerror:" in lower:
        return "BASELINE_OR_TEST_BUG", compact(text)
    if "tensor-likes are not close" in lower:
        return "NUMERICAL_MISMATCH", compact(text)
    if status == "FAILED_PARITY":
        return "PARITY_ERROR", compact(text)
    return "UNKNOWN", compact(text)


def load_results(results_dir: pathlib.Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            rows.append(
                ResultRow(
                    path=path,
                    data={"status": "UNREADABLE"},
                    status="UNREADABLE",
                    family=path.stem,
                    category="UNREADABLE",
                    excerpt=str(exc),
                )
            )
            continue
        status = str(data.get("status", "<missing>"))
        family = family_from_entry(data, path)
        category, excerpt = classify(data)
        rows.append(ResultRow(path=path, data=data, status=status, family=family, category=category, excerpt=excerpt))
    return rows


def rank_category(category: str) -> int:
    return ACTIONABLE_ORDER.get(category, ACTIONABLE_ORDER["UNKNOWN"])


def summarize(rows: Iterable[ResultRow]) -> dict:
    summary = {
        "status_counts": collections.Counter(),
        "category_counts": collections.Counter(),
        "family_counts": collections.Counter(),
        "by_category": collections.defaultdict(list),
        "approved": [],
    }
    for row in rows:
        summary["status_counts"][row.status] += 1
        summary["category_counts"][row.category] += 1
        summary["family_counts"][row.family] += 1
        if row.status == "APPROVED":
            summary["approved"].append(row)
            continue
        summary["by_category"][row.category].append(row)
    return summary


def top_families(rows: list[ResultRow], limit: int) -> list[tuple[str, int]]:
    counts = collections.Counter(row.family for row in rows)
    return counts.most_common(limit)


def format_markdown(summary: dict, rows: list[ResultRow], limit: int, family_limit: int) -> str:
    lines: list[str] = []
    lines.append("# Queue Triage")
    lines.append("")
    lines.append("## Status Counts")
    for status, count in summary["status_counts"].most_common():
        lines.append(f"- `{status}`: {count}")
    lines.append("")
    lines.append("## Failure Categories")
    ordered_categories = sorted(
        (c for c in summary["category_counts"] if c != "APPROVED"),
        key=lambda c: (rank_category(c), -summary["category_counts"][c], c),
    )
    for category in ordered_categories:
        count = summary["category_counts"][category]
        lines.append(f"### `{category}` ({count})")
        cat_rows = summary["by_category"].get(category, [])
        for family, fam_count in top_families(cat_rows, family_limit):
            lines.append(f"- `{family}`: {fam_count}")
        for row in cat_rows[:limit]:
            rel = row.path.relative_to(ROOT)
            excerpt = row.excerpt or "(no excerpt)"
            lines.append(f"- `{rel}` -> `{row.family}`")
            lines.append(f"  - {excerpt}")
        lines.append("")

    lines.append("## Approved Kernels")
    for row in summary["approved"]:
        rel = row.path.relative_to(ROOT)
        lines.append(f"- `{rel}`")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def print_report(summary: dict, rows: list[ResultRow], limit: int, family_limit: int) -> None:
    print("QUEUE TRIAGE")
    print("=" * 60)
    print("Status counts:")
    for status, count in summary["status_counts"].most_common():
        print(f"  {status}: {count}")

    print("\nMost actionable failure categories:")
    ordered_categories = sorted(
        (c for c in summary["category_counts"] if c not in {"APPROVED"}),
        key=lambda c: (rank_category(c), -summary["category_counts"][c], c),
    )
    for category in ordered_categories:
        count = summary["category_counts"][category]
        print(f"\n[{category}] {count}")
        cat_rows = summary["by_category"].get(category, [])
        for family, fam_count in top_families(cat_rows, family_limit):
            print(f"  {family}: {fam_count}")
        for row in cat_rows[:limit]:
            print(f"  - {row.path.name} -> {row.family}")
            if row.excerpt:
                print(f"    {row.excerpt}")

    if summary["approved"]:
        print("\nApproved kernels:")
        for row in summary["approved"]:
            print(f"  - {row.path.name} -> {row.family}")


def main() -> int:
    args = parse_args()
    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.exists():
        print(f"error: results dir not found: {results_dir}", file=sys.stderr)
        return 2

    rows = load_results(results_dir)
    summary = summarize(rows)
    print_report(summary, rows, args.limit, args.families)

    if not args.no_markdown:
        markdown = format_markdown(summary, rows, args.limit, args.families)
        md_path = pathlib.Path(args.markdown)
        if not md_path.is_absolute():
            md_path = (ROOT / md_path).resolve()
        md_path.write_text(markdown)
        print(f"\nMarkdown report written to {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
