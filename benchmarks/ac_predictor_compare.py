import importlib.util
import pathlib
import subprocess
import sys
import tempfile

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ac_predictor import VisionTransformerPredictorAC


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


def benchmark_cuda(fn, warmup=10, iters=40):
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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    baseline = load_module_from_head(ROOT, "src/models/ac_predictor.py", "baseline_src_ac_predictor_bench")
    def make_case(use_extrinsics):
        kwargs = dict(
            img_size=128,
            patch_size=16,
            num_frames=4,
            tubelet_size=1,
            embed_dim=256,
            predictor_embed_dim=256,
            depth=4,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            use_rope=True,
            action_embed_dim=7,
            use_extrinsics=use_extrinsics,
        )
        optimized = VisionTransformerPredictorAC(**kwargs).cuda().half().eval()
        reference = baseline.VisionTransformerPredictorAC(**kwargs).cuda().half().eval()
        reference.load_state_dict(optimized.state_dict())

        B = 4
        T = kwargs["num_frames"] // kwargs["tubelet_size"]
        tokens = T * ((kwargs["img_size"] // kwargs["patch_size"]) ** 2)
        x = torch.randn(B, tokens, kwargs["embed_dim"], device="cuda", dtype=torch.float16)
        actions = torch.randn(B, T, kwargs["action_embed_dim"], device="cuda", dtype=torch.float16)
        states = torch.randn(B, T, kwargs["action_embed_dim"], device="cuda", dtype=torch.float16)
        extrinsics = None
        if use_extrinsics:
            extrinsics = torch.randn(B, T, kwargs["action_embed_dim"] - 1, device="cuda", dtype=torch.float16)

        def baseline_fn():
            with torch.no_grad():
                return reference(x, actions, states, extrinsics=extrinsics)

        def optimized_fn():
            with torch.no_grad():
                return optimized(x, actions, states, extrinsics=extrinsics)

        with torch.no_grad():
            torch.testing.assert_close(optimized_fn(), baseline_fn(), atol=3e-3, rtol=3e-3)

        baseline_ms = benchmark_cuda(baseline_fn)
        optimized_ms = benchmark_cuda(optimized_fn)
        return {
            "name": "ac_predictor_forward_extrinsics" if use_extrinsics else "ac_predictor_forward",
            "baseline_ms": baseline_ms,
            "optimized_ms": optimized_ms,
            "speedup_pct": ((baseline_ms - optimized_ms) / baseline_ms) * 100.0,
        }

    for row in [make_case(False), make_case(True)]:
        print(row)


if __name__ == "__main__":
    main()
