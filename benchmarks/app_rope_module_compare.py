import importlib.util
import pathlib
import subprocess
import sys
import tempfile

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vjepa_2_1.models.utils.modules import Block, RoPEAttention, rotate_query_key_pair, rotate_queries_or_keys


def load_module_from_head(repo_root, git_path, module_name):
    content = subprocess.check_output(["git", "show", f"HEAD:{git_path}"], cwd=repo_root, text=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(content)
        temp_path = handle.name
    spec = importlib.util.spec_from_file_location(module_name, temp_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def benchmark_cuda(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _make_attn_kwargs():
    return dict(
        dim=256,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=8,
        is_causal=False,
        n_registers=4,
        has_cls_first=True,
        interpolate_rope=False,
        patch_size=16,
    )


def compare_pair_helper(warmup=40, iters=400):
    q = torch.randn(8, 16, 1029, 24, device="cuda", dtype=torch.float16)
    k = torch.randn(8, 16, 1029, 24, device="cuda", dtype=torch.float16)
    pos = torch.arange(1024, device="cuda", dtype=torch.float16)

    def baseline_fn():
        baseline_q = rotate_queries_or_keys(q, pos, n_registers=4, has_cls_first=True)
        baseline_k = rotate_queries_or_keys(k, pos, n_registers=4, has_cls_first=True)
        return baseline_q, baseline_k

    def optimized_fn():
        return rotate_query_key_pair(q, k, pos, n_registers=4, has_cls_first=True)

    baseline_q, baseline_k = baseline_fn()
    optimized_q, optimized_k = optimized_fn()
    torch.testing.assert_close(optimized_q, baseline_q, atol=0, rtol=0)
    torch.testing.assert_close(optimized_k, baseline_k, atol=0, rtol=0)
    return {
        "name": "rotate_query_key_pair",
        "baseline_ms": benchmark_cuda(baseline_fn, warmup, iters),
        "optimized_ms": benchmark_cuda(optimized_fn, warmup, iters),
    }


def compare_rope_attention(baseline_modules, warmup=20, iters=100):
    kwargs = _make_attn_kwargs()
    optimized = RoPEAttention(**kwargs).cuda().half().eval()
    baseline = baseline_modules.RoPEAttention(**kwargs).cuda().half().eval()
    baseline.load_state_dict(optimized.state_dict())

    B, T, H, W, C = 2, 4, 8, 8, 256
    x = torch.randn(B, 1 + (T * H * W) + 4, C, device="cuda", dtype=torch.float16)

    def baseline_fn():
        with torch.no_grad():
            return baseline(x, T=T, H_patches=H, W_patches=W)[0]

    def optimized_fn():
        with torch.no_grad():
            return optimized(x, T=T, H_patches=H, W_patches=W)[0]

    torch.testing.assert_close(optimized_fn(), baseline_fn(), atol=0, rtol=0)
    return {
        "name": "rope_attention_forward",
        "baseline_ms": benchmark_cuda(baseline_fn, warmup, iters),
        "optimized_ms": benchmark_cuda(optimized_fn, warmup, iters),
    }


def compare_rope_block(baseline_modules, warmup=20, iters=100):
    kwargs = _make_attn_kwargs()
    kwargs.update(
        dict(
            use_rope=True,
            mlp_ratio=2.0,
            drop=0.0,
            drop_path=0.0,
            act_layer=torch.nn.GELU,
            wide_silu=True,
        )
    )
    optimized = Block(**kwargs).cuda().half().eval()
    baseline = baseline_modules.Block(**kwargs).cuda().half().eval()
    baseline.load_state_dict(optimized.state_dict())

    B, T, H, W, C = 2, 4, 8, 8, 256
    x = torch.randn(B, 1 + (T * H * W) + 4, C, device="cuda", dtype=torch.float16)

    def baseline_fn():
        with torch.no_grad():
            return baseline(x, T=T, H_patches=H, W_patches=W)[0]

    def optimized_fn():
        with torch.no_grad():
            return optimized(x, T=T, H_patches=H, W_patches=W)[0]

    torch.testing.assert_close(optimized_fn(), baseline_fn(), atol=0, rtol=0)
    return {
        "name": "rope_block_forward",
        "baseline_ms": benchmark_cuda(baseline_fn, warmup, iters),
        "optimized_ms": benchmark_cuda(optimized_fn, warmup, iters),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    baseline_modules = load_module_from_head(ROOT, "app/vjepa_2_1/models/utils/modules.py", "baseline_app_rope_modules_bench")
    results = [
        compare_pair_helper(),
        compare_rope_attention(baseline_modules),
        compare_rope_block(baseline_modules),
    ]
    for row in results:
        row["speedup_pct"] = ((row["baseline_ms"] - row["optimized_ms"]) / row["baseline_ms"]) * 100.0
        print(row)


if __name__ == "__main__":
    main()
