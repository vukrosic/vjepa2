import importlib.util
import pathlib
import subprocess
import sys
import tempfile

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vjepa_2_1.models.predictor import VisionTransformerPredictor


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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    baseline = load_module_from_head(ROOT, "app/vjepa_2_1/models/predictor.py", "baseline_app_predictor_bench")
    kwargs = dict(
        img_size=256,
        patch_size=16,
        num_frames=1,
        embed_dim=256,
        predictor_embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_rope=True,
        n_output_distillation=1,
        modality_embedding=False,
    )
    optimized = VisionTransformerPredictor(**kwargs).cuda().half().eval()
    reference = baseline.VisionTransformerPredictor(**kwargs).cuda().half().eval()
    reference.load_state_dict(optimized.state_dict())

    B, N_ctxt = 8, 64
    x = torch.randn(B, N_ctxt, kwargs["embed_dim"], device="cuda", dtype=torch.float16)
    num_patches = (kwargs["img_size"] // kwargs["patch_size"]) ** 2
    perms = torch.stack(
        [torch.randperm(num_patches, device="cuda", dtype=torch.long) for _ in range(B)],
        dim=0,
    )
    masks_x = [torch.sort(perms[:, :N_ctxt], dim=1).values]
    masks_y = [torch.sort(perms[:, N_ctxt : 2 * N_ctxt], dim=1).values]

    def baseline_fn():
        return reference(x, masks_x, masks_y)[0]

    def optimized_fn():
        return optimized(x, masks_x, masks_y)[0]

    torch.testing.assert_close(optimized_fn(), baseline_fn(), atol=2e-3, rtol=2e-3)

    baseline_ms = benchmark_cuda(baseline_fn)
    optimized_ms = benchmark_cuda(optimized_fn)
    print(
        {
            "name": "app_predictor_forward",
            "baseline_ms": baseline_ms,
            "optimized_ms": optimized_ms,
            "speedup_pct": ((baseline_ms - optimized_ms) / baseline_ms) * 100.0,
        }
    )


if __name__ == "__main__":
    main()
