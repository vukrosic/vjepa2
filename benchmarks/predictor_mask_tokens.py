import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import time

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.predictor import VisionTransformerPredictor


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


def benchmark_cuda(model, enc, masks_x, masks_y, warmup=20, iters=100):
    for _ in range(warmup):
        model(enc, masks_x, masks_y)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        model(enc, masks_x, masks_y)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_cpu(model, enc, masks_x, masks_y, warmup=10, iters=50):
    for _ in range(warmup):
        model(enc, masks_x, masks_y)
    start = time.perf_counter()
    for _ in range(iters):
        model(enc, masks_x, masks_y)
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def main():
    baseline = load_module_from_head(ROOT, "src/models/predictor.py", "baseline_predictor_mask_tokens")
    kwargs = dict(
        img_size=256,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=1408,
        predictor_embed_dim=384,
        depth=2,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        use_mask_tokens=True,
        num_mask_tokens=10,
        zero_init_mask_tokens=True,
        use_rope=True,
    )
    current = VisionTransformerPredictor(**kwargs).eval()
    reference = baseline.VisionTransformerPredictor(**kwargs).eval()
    reference.load_state_dict(current.state_dict())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    current = current.to(device)
    reference = reference.to(device)

    B = 2
    num_context = 256
    num_target = 256
    enc_mask_indices = [torch.arange(num_context, device=device).unsqueeze(0).repeat(B, 1)]
    target_mask_indices = [torch.arange(num_context, num_context + num_target, device=device).unsqueeze(0).repeat(B, 1)]
    enc = torch.randn(B, num_context, kwargs["embed_dim"], device=device)

    with torch.no_grad():
        current_out = current(enc, enc_mask_indices, target_mask_indices)
        reference_out = reference(enc, enc_mask_indices, target_mask_indices)
    torch.testing.assert_close(current_out, reference_out, atol=1e-4, rtol=1e-4)

    with torch.no_grad():
        if device == "cuda":
            baseline_ms = benchmark_cuda(reference, enc, enc_mask_indices, target_mask_indices)
            current_ms = benchmark_cuda(current, enc, enc_mask_indices, target_mask_indices)
        else:
            baseline_ms = benchmark_cpu(reference, enc, enc_mask_indices, target_mask_indices)
            current_ms = benchmark_cpu(current, enc, enc_mask_indices, target_mask_indices)

    print(f"device: {device}")
    print(f"baseline_ms: {baseline_ms:.4f}")
    print(f"current_ms: {current_ms:.4f}")
    print(f"speedup_x: {baseline_ms / current_ms:.4f}")


if __name__ == "__main__":
    main()
