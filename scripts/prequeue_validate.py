#!/usr/bin/env python3
"""Fast prequeue validation for kernel experiments.

This helper is intentionally conservative:
- compile all provided files for syntax
- import kernel/test modules to catch import-time failures
- optionally run a single pytest parity pass for the test file

Benchmark files are syntax-checked only by default because most of them are
executable scripts with top-level benchmark code.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Iterable, Optional


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_KERNELS_DIR = ROOT / "src/models/utils/kernels"
DEFAULT_TESTS_DIR = ROOT / "tests/queue"
DEFAULT_BENCHS_DIR = ROOT / "benchmarks/queue"


EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_MISSING_PATH = 3
EXIT_SMOKE_FAIL = 4
EXIT_PARITY_FAIL = 5


@dataclass
class Check:
    label: str
    path: pathlib.Path
    ok: bool
    output: str = ""


def _run(cmd: list[str], *, cwd: pathlib.Path = ROOT, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _print_block(title: str, output: str) -> None:
    print(f"\n[{title}]")
    print(output.rstrip() or "(no output)")


def _resolve_path(path_arg: Optional[str], default_path: pathlib.Path) -> pathlib.Path:
    if path_arg:
        path = pathlib.Path(path_arg).expanduser()
        return path if path.is_absolute() else (ROOT / path).resolve()
    return default_path


def _compile_syntax(path: pathlib.Path, label: str) -> Check:
    if not path.exists():
        return Check(label=label, path=path, ok=False, output=f"missing file: {path}")
    proc = _run([sys.executable, "-m", "py_compile", str(path)])
    output = proc.stdout + proc.stderr
    return Check(label=label, path=path, ok=proc.returncode == 0, output=output)


def _import_module(path: pathlib.Path, label: str) -> Check:
    if not path.exists():
        return Check(label=label, path=path, ok=False, output=f"missing file: {path}")

    code = textwrap.dedent(
        """
        import importlib.util
        import pathlib
        import types
        import sys

        root = pathlib.Path(sys.argv[1])
        path = pathlib.Path(sys.argv[2])
        sys.path.insert(0, str(root))

        if "pytest" not in sys.modules:
            class _Mark:
                def __getattr__(self, name):
                    def factory(*args, **kwargs):
                        def decorator(fn):
                            return fn
                        return decorator
                    return factory

            pytest_stub = types.ModuleType("pytest")
            pytest_stub.mark = _Mark()
            pytest_stub.fixture = lambda *args, **kwargs: (lambda fn: fn)
            pytest_stub.skip = lambda *args, **kwargs: None
            sys.modules["pytest"] = pytest_stub

        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        print(f"import ok: {path}")
        """
    )
    proc = _run([sys.executable, "-c", code, str(ROOT), str(path)])
    output = proc.stdout + proc.stderr
    return Check(label=label, path=path, ok=proc.returncode == 0, output=output)


def _run_parity(test_path: pathlib.Path) -> Check:
    if not test_path.exists():
        return Check(label="parity", path=test_path, ok=False, output=f"missing file: {test_path}")

    if importlib.util.find_spec("pytest") is None:
        return Check(
            label="parity",
            path=test_path,
            ok=False,
            output="pytest is not installed in this environment; parity step cannot run",
        )

    proc = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_path),
            "-q",
            "-x",
            "--maxfail=1",
            "--tb=short",
            "--no-header",
        ],
        timeout=180,
    )
    output = proc.stdout + proc.stderr
    return Check(label="parity", path=test_path, ok=proc.returncode == 0, output=output)


def _run_benchmark(bench_path: pathlib.Path) -> Check:
    if not bench_path.exists():
        return Check(label="bench", path=bench_path, ok=False, output=f"missing file: {bench_path}")

    proc = _run([sys.executable, str(bench_path)], timeout=300)
    output = proc.stdout + proc.stderr
    ok = proc.returncode == 0 and "BENCH_RESULT=" in proc.stdout
    if proc.returncode == 0 and "BENCH_RESULT=" not in proc.stdout:
        output = (output.rstrip() + "\nmissing BENCH_RESULT=... output").strip()
    return Check(label="bench", path=bench_path, ok=ok, output=output)


def _default_paths(kernel: Optional[str]) -> tuple[Optional[pathlib.Path], Optional[pathlib.Path], Optional[pathlib.Path]]:
    if not kernel:
        return None, None, None
    return (
        (DEFAULT_KERNELS_DIR / f"{kernel}.py").resolve(),
        (DEFAULT_TESTS_DIR / f"test_{kernel}.py").resolve(),
        (DEFAULT_BENCHS_DIR / f"bench_{kernel}.py").resolve(),
    )


def _validate_inputs(paths: Iterable[pathlib.Path]) -> Optional[str]:
    for path in paths:
        if path is not None and not path.exists():
            return f"missing file: {path}"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a kernel experiment before enqueueing it.",
    )
    parser.add_argument("--kernel", help="Kernel name, e.g. fused_gelu_linear")
    parser.add_argument("--kernel-path", help="Explicit kernel file path")
    parser.add_argument("--test-path", help="Explicit parity test path")
    parser.add_argument("--bench-path", help="Explicit benchmark path")
    parser.add_argument(
        "--parity",
        action="store_true",
        help="Run a single pytest parity pass if a test file exists.",
    )
    parser.add_argument(
        "--run-parity",
        action="store_true",
        dest="parity",
        help="Alias for --parity.",
    )
    parser.add_argument(
        "--run-bench",
        action="store_true",
        help="Run the benchmark script and require BENCH_RESULT output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    kernel_default, test_default, bench_default = _default_paths(args.kernel)

    kernel_path = _resolve_path(args.kernel_path, kernel_default) if kernel_default or args.kernel_path else None
    test_path = _resolve_path(args.test_path, test_default) if test_default or args.test_path else None
    bench_path = _resolve_path(args.bench_path, bench_default) if bench_default or args.bench_path else None

    if not any([kernel_path, test_path, bench_path]):
        print("error: provide --kernel or at least one of --kernel-path/--test-path/--bench-path", file=sys.stderr)
        return EXIT_BAD_ARGS

    missing = _validate_inputs([p for p in [kernel_path, test_path, bench_path] if p is not None])
    if missing:
        print(f"error: {missing}", file=sys.stderr)
        return EXIT_MISSING_PATH

    checks: list[Check] = []

    if kernel_path is not None:
        checks.append(_compile_syntax(kernel_path, "kernel syntax"))
        checks.append(_import_module(kernel_path, "kernel import"))

    if test_path is not None:
        checks.append(_compile_syntax(test_path, "test syntax"))
        checks.append(_import_module(test_path, "test import"))

    if bench_path is not None:
        checks.append(_compile_syntax(bench_path, "bench syntax"))

    failed = [c for c in checks if not c.ok]
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(f"{status}: {check.label} -> {check.path}")
        if not check.ok:
            _print_block(check.label, check.output)

    if failed:
        return EXIT_SMOKE_FAIL

    if args.parity and test_path is not None:
        parity = _run_parity(test_path)
        status = "OK" if parity.ok else "FAIL"
        print(f"{status}: parity -> {parity.path}")
        _print_block("parity", parity.output)
        if not parity.ok:
            return EXIT_PARITY_FAIL
    elif args.parity:
        print("skipping parity: no test path available")

    if args.run_bench and bench_path is not None:
        bench = _run_benchmark(bench_path)
        status = "OK" if bench.ok else "FAIL"
        print(f"{status}: bench -> {bench.path}")
        _print_block("bench", bench.output)
        if not bench.ok:
            return EXIT_PARITY_FAIL
    elif args.run_bench:
        print("skipping bench: no benchmark path available")

    print("\nprequeue validation passed")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
