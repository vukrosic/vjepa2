# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import pytest
import torch

from tests.support.kernel_benchmark import (
    AttentionCase,
    KernelResult,
    benchmark_policy,
    default_attention_cases,
    default_policies,
    format_speedup_table,
    summarize_speedups,
)


def test_default_policies_include_baseline():
    assert default_policies() == ["optimized", "default_auto", "math_only"]


def test_speedup_summary_and_table_formatting():
    rows = summarize_speedups(
        [
            KernelResult(
                case="unmasked",
                policy="default_auto",
                masked=False,
                dtype="float16",
                shape=(8, 16, 1024, 64),
                backend_order=("DEFAULT_AUTO",),
                median_ms=2.0,
                mean_ms=2.1,
                min_ms=1.9,
                samples_ms=(2.0,),
            ),
            KernelResult(
                case="unmasked",
                policy="optimized",
                masked=False,
                dtype="float16",
                shape=(8, 16, 1024, 64),
                backend_order=("FLASH_ATTENTION", "MATH"),
                median_ms=1.0,
                mean_ms=1.0,
                min_ms=1.0,
                samples_ms=(1.0,),
            ),
        ]
    )
    summary = defaultdict(dict)
    for row in rows:
        summary[row["case"]][row["policy"]] = row

    assert summary["unmasked"]["optimized"]["speedup_x"] == pytest.approx(2.0)
    assert summary["unmasked"]["optimized"]["delta_pct"] == pytest.approx(50.0)
    table = format_speedup_table(rows)
    assert "| case | policy | median_ms | baseline_ms | speedup_x | delta_pct | backend_order |" in table
    assert "1.000x" in table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires CUDA")
def test_cuda_benchmark_policy_reports_backend_order():
    case = AttentionCase(name="unmasked", shape=(1, 2, 64, 16), dtype=torch.float16, masked=False)
    result = benchmark_policy(case, policy="optimized", warmup_iters=2, timed_iters=4, repeats=2, check_parity=True)
    assert result.policy == "optimized"
    assert result.backend_order[0] == "FLASH_ATTENTION"
    assert result.median_ms > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires CUDA")
def test_cuda_benchmark_policy_skips_flash_for_masked_case():
    case = AttentionCase(name="masked", shape=(1, 2, 64, 16), dtype=torch.float16, masked=True)
    result = benchmark_policy(case, policy="optimized", warmup_iters=2, timed_iters=4, repeats=2, check_parity=True)
    assert "FLASH_ATTENTION" not in result.backend_order
    assert result.backend_order[0] == "EFFICIENT_ATTENTION"


def test_default_attention_cases_build_expected_cases():
    cases = default_attention_cases(8, 16, 1024, 64, "fp16")
    assert [case.name for case in cases] == ["unmasked", "masked"]
