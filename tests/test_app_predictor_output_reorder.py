# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


def _make_kwargs():
    return dict(
        img_size=64,
        patch_size=16,
        num_frames=1,
        embed_dim=96,
        predictor_embed_dim=96,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_rope=True,
        n_output_distillation=1,
        modality_embedding=False,
    )


def test_app_predictor_output_reorder_matches_head_baseline_cpu():
    baseline = load_module_from_head(ROOT, "app/vjepa_2_1/models/predictor.py", "baseline_app_predictor_reorder")
    kwargs = _make_kwargs()

    optimized = VisionTransformerPredictor(**kwargs).eval()
    reference = baseline.VisionTransformerPredictor(**kwargs).eval()
    reference.load_state_dict(optimized.state_dict())

    B, ctxt_tokens, target_tokens = 3, 5, 5
    num_patches = (kwargs["img_size"] // kwargs["patch_size"]) ** 2
    generator = torch.Generator().manual_seed(1234)
    perms = torch.stack(
        [torch.randperm(num_patches, generator=generator, dtype=torch.int64) for _ in range(B)],
        dim=0,
    )
    enc_mask_indices = [torch.sort(perms[:, :ctxt_tokens], dim=1).values]
    target_mask_indices = [torch.sort(perms[:, ctxt_tokens : ctxt_tokens + target_tokens], dim=1).values]
    enc = torch.randn(B, ctxt_tokens, kwargs["embed_dim"], generator=generator)

    with torch.no_grad():
        optimized_out, _ = optimized(enc, enc_mask_indices, target_mask_indices)
        reference_out, _ = reference(enc, enc_mask_indices, target_mask_indices)

    torch.testing.assert_close(optimized_out, reference_out, atol=1e-6, rtol=1e-6)


def test_app_predictor_output_reorder_duplicate_masks_falls_back_to_head_cpu():
    baseline = load_module_from_head(ROOT, "app/vjepa_2_1/models/predictor.py", "baseline_app_predictor_reorder_dupes")
    kwargs = _make_kwargs()

    optimized = VisionTransformerPredictor(**kwargs).eval()
    reference = baseline.VisionTransformerPredictor(**kwargs).eval()
    reference.load_state_dict(optimized.state_dict())

    enc_mask_indices = [torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.int64)]
    target_mask_indices = [torch.tensor([[2, 2, 3], [0, 0, 3]], dtype=torch.int64)]
    enc = torch.randn(2, 3, kwargs["embed_dim"])

    with torch.no_grad():
        optimized_out, _ = optimized(enc, enc_mask_indices, target_mask_indices)
        reference_out, _ = reference(enc, enc_mask_indices, target_mask_indices)

    torch.testing.assert_close(optimized_out, reference_out, atol=1e-6, rtol=1e-6)
