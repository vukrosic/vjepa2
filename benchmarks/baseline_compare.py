import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.utils.modules import (
    ACRoPEAttention,
    Attention,
    RoPEAttention,
    build_action_block_causal_attention_mask,
    rotate_queries_or_keys,
)
from src.models.attentive_pooler import AttentivePooler
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch


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


def benchmark_cpu(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def benchmark_cuda(fn, warmup, iters):
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


def format_rows_markdown(rows):
    header = ["name", "baseline_ms", "optimized_ms", "speedup_pct"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    f'{row["baseline_ms"]:.4f}',
                    f'{row["optimized_ms"]:.4f}',
                    f'{row["speedup_pct"]:.2f}%',
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def compare_repeat_interleave(baseline_tensors, warmup=20, iters=200):
    x = torch.randn(2048, 1024, device="cuda", dtype=torch.float16)
    B, repeat = 32, 4
    baseline_fn = lambda: baseline_tensors.repeat_interleave_batch(x, B, repeat)
    optimized_fn = lambda: repeat_interleave_batch(x, B, repeat)
    torch.testing.assert_close(optimized_fn(), baseline_fn())
    return {
        "name": "repeat_interleave_batch",
        "baseline_ms": benchmark_cuda(baseline_fn, warmup, iters),
        "optimized_ms": benchmark_cuda(optimized_fn, warmup, iters),
    }


def compare_apply_masks(warmup=20, iters=200):
    x = torch.randn(32, 1024, 384, device="cuda", dtype=torch.float16)
    masks = [
        torch.randint(0, 1024, (32, 256), device="cuda", dtype=torch.long),
        torch.randint(0, 1024, (32, 256), device="cuda", dtype=torch.long),
    ]

    def baseline():
        all_x = []
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
        return torch.cat(all_x, dim=0)

    def optimized():
        return apply_masks(x, masks)

    torch.testing.assert_close(optimized(), baseline())
    return {
        "name": "apply_masks",
        "baseline_ms": benchmark_cuda(baseline, warmup, iters),
        "optimized_ms": benchmark_cuda(optimized, warmup, iters),
    }


def compare_action_mask(warmup=20, iters=500):
    baseline_fn = lambda: _baseline_action_block_causal_attention_mask(8, 14, 14, 2)
    optimized_fn = lambda: build_action_block_causal_attention_mask(8, 14, 14, 2)
    return {
        "name": "build_action_block_causal_attention_mask",
        "baseline_ms": benchmark_cpu(baseline_fn, warmup, iters),
        "optimized_ms": benchmark_cpu(optimized_fn, warmup, iters),
    }


def compare_attention_block(warmup=20, iters=100):
    current = Attention(dim=384, num_heads=6, proj_drop=0.0, attn_drop=0.0, use_sdpa=True).cuda().half().eval()
    baseline_modules = load_module_from_head(ROOT, "src/models/utils/modules.py", "baseline_src_models_utils_modules")
    baseline = baseline_modules.Attention(dim=384, num_heads=6, proj_drop=0.0, attn_drop=0.0, use_sdpa=True).cuda().half().eval()
    baseline.load_state_dict(current.state_dict())

    x = torch.randn(4, 256, 384, device="cuda", dtype=torch.float16)

    def baseline_fn():
        return baseline(x)

    def optimized_fn():
        return current(x)

    torch.testing.assert_close(optimized_fn(), baseline_fn(), atol=2e-3, rtol=2e-3)
    return {
        "name": "attention_block_forward",
        "baseline_ms": benchmark_cuda(baseline_fn, warmup, iters),
        "optimized_ms": benchmark_cuda(optimized_fn, warmup, iters),
    }


def compare_ac_rope_attention(baseline_modules, warmup=20, iters=100):
    kwargs = dict(dim=256, num_heads=8, qkv_bias=True, proj_drop=0.0, attn_drop=0.0, use_sdpa=False, grid_size=8)
    optimized = ACRoPEAttention(**kwargs).cuda().eval()
    baseline = baseline_modules.ACRoPEAttention(**kwargs).cuda().eval()
    baseline.load_state_dict(optimized.state_dict())

    B, T, H, W, A, C = 2, 4, 8, 8, 2, 256
    x = torch.randn(B, T * (A + H * W), C, device="cuda")

    def baseline_fn():
        return baseline(x, T=T, H=H, W=W, action_tokens=A)

    def optimized_fn():
        return optimized(x, T=T, H=H, W=W, action_tokens=A)

    torch.testing.assert_close(optimized_fn(), baseline_fn(), atol=1e-5, rtol=1e-5)
    return {
        "name": "ac_rope_attention_forward",
        "baseline_ms": benchmark_cuda(baseline_fn, warmup, iters),
        "optimized_ms": benchmark_cuda(optimized_fn, warmup, iters),
    }


def compare_rope_attention(baseline_modules, warmup=20, iters=100):
    kwargs = dict(dim=256, num_heads=8, qkv_bias=True, proj_drop=0.0, attn_drop=0.0, use_sdpa=False, grid_size=8)
    optimized = RoPEAttention(**kwargs).cuda().eval()
    baseline = baseline_modules.RoPEAttention(**kwargs).cuda().eval()
    baseline.load_state_dict(optimized.state_dict())

    B, T, H, W, C = 2, 4, 8, 8, 256
    x = torch.randn(B, T * H * W, C, device="cuda")

    def baseline_fn():
        return baseline(x, T=T, H_patches=H, W_patches=W)

    def optimized_fn():
        return optimized(x, T=T, H_patches=H, W_patches=W)

    torch.testing.assert_close(optimized_fn(), baseline_fn(), atol=1e-5, rtol=1e-5)
    return {
        "name": "rope_attention_forward",
        "baseline_ms": benchmark_cuda(baseline_fn, warmup, iters),
        "optimized_ms": benchmark_cuda(optimized_fn, warmup, iters),
    }


def compare_attentive_pooler(warmup=20, iters=100):
    baseline_pooler = load_module_from_head(ROOT, "src/models/attentive_pooler.py", "baseline_src_attentive_pooler")
    kwargs = dict(
        num_queries=4,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=2.0,
        depth=2,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
    )
    optimized = AttentivePooler(**kwargs).cuda().half().eval()
    baseline = baseline_pooler.AttentivePooler(**kwargs).cuda().half().eval()
    baseline.load_state_dict(optimized.state_dict())

    x = torch.randn(8, 256, 384, device="cuda", dtype=torch.float16)

    def baseline_fn():
        return baseline(x)

    def optimized_fn():
        return optimized(x)

    torch.testing.assert_close(optimized_fn(), baseline_fn(), atol=2e-3, rtol=2e-3)
    return {
        "name": "attentive_pooler_forward",
        "baseline_ms": benchmark_cuda(baseline_fn, warmup, iters),
        "optimized_ms": benchmark_cuda(optimized_fn, warmup, iters),
    }


def compare_rotate_queries_or_keys(warmup=20, iters=200):
    x = torch.randn(8, 16, 1024, 64, device="cuda", dtype=torch.float16)
    pos = torch.arange(1024, device="cuda", dtype=torch.float16).view(1, 1, 1024)

    def baseline():
        return _baseline_rotate_queries_or_keys(x, pos)

    def optimized():
        return rotate_queries_or_keys(x, pos)

    torch.testing.assert_close(optimized(), baseline(), atol=2e-3, rtol=2e-3)
    return {
        "name": "rotate_queries_or_keys",
        "baseline_ms": benchmark_cuda(baseline, warmup, iters),
        "optimized_ms": benchmark_cuda(optimized, warmup, iters),
    }


def _baseline_action_block_causal_attention_mask(T, H, W, add_tokens=1):
    tokens_per_step = add_tokens + (H * W)
    total_tokens = T * tokens_per_step
    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool)
    block = torch.ones(tokens_per_step, tokens_per_step, dtype=torch.bool)
    for t1 in range(T):
        for t2 in range(max(0, t1 - T + 1), t1 + 1):
            mask[t1 * tokens_per_step : (t1 + 1) * tokens_per_step, t2 * tokens_per_step : (t2 + 1) * tokens_per_step] = block
    return mask


def _baseline_rotate_queries_or_keys(x, pos):
    _, _, _, D = x.size()
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freq.sin()
    emb_cos = freq.cos()
    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    baseline_tensors = load_module_from_head(ROOT, "src/utils/tensors.py", "baseline_src_tensors")
    baseline_modules = load_module_from_head(ROOT, "src/models/utils/modules.py", "baseline_src_models_utils_modules")

    rows = [
        compare_action_mask(),
        compare_attention_block(),
        compare_attentive_pooler(),
        compare_rotate_queries_or_keys(),
        compare_repeat_interleave(baseline_tensors),
        compare_apply_masks(),
        compare_ac_rope_attention(baseline_modules),
        compare_rope_attention(baseline_modules),
    ]

    for row in rows:
        row["speedup_pct"] = ((row["baseline_ms"] - row["optimized_ms"]) / row["baseline_ms"]) * 100.0

    print(format_rows_markdown(rows))
    print()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
