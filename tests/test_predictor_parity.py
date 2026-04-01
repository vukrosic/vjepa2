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

from src.models.predictor import VisionTransformerPredictor, _get_precomputed_rope_positions, _get_sorted_target_positions


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


def test_src_predictor_matches_head_baseline_cpu():
    baseline = load_module_from_head(ROOT, "src/models/predictor.py", "baseline_src_predictor")

    kwargs = dict(
        img_size=32,
        patch_size=16,
        num_frames=1,
        embed_dim=64,
        predictor_embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_rope=False,
    )
    optimized = VisionTransformerPredictor(**kwargs).eval()
    reference = baseline.VisionTransformerPredictor(**kwargs).eval()
    reference.load_state_dict(optimized.state_dict())

    B = 2
    enc_mask_indices = [torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)]
    target_mask_indices = [torch.tensor([[2, 3], [0, 3]], dtype=torch.int64)]
    enc = torch.randn(B, enc_mask_indices[0].size(1), 64)

    with torch.no_grad():
        optimized_out = optimized(enc, enc_mask_indices, target_mask_indices)
        reference_out = reference(enc, enc_mask_indices, target_mask_indices)

    torch.testing.assert_close(optimized_out, reference_out, atol=1e-5, rtol=1e-5)


def test_src_predictor_rope_matches_head_baseline_cpu():
    baseline = load_module_from_head(ROOT, "src/models/predictor.py", "baseline_src_predictor_rope")

    kwargs = dict(
        img_size=32,
        patch_size=16,
        num_frames=1,
        embed_dim=96,
        predictor_embed_dim=96,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_rope=True,
    )
    optimized = VisionTransformerPredictor(**kwargs).eval()
    reference = baseline.VisionTransformerPredictor(**kwargs).eval()
    reference.load_state_dict(optimized.state_dict())

    B = 2
    enc_mask_indices = [torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)]
    target_mask_indices = [torch.tensor([[2, 3], [0, 3]], dtype=torch.int64)]
    enc = torch.randn(B, enc_mask_indices[0].size(1), 96)

    with torch.no_grad():
        optimized_out = optimized(enc, enc_mask_indices, target_mask_indices)
        reference_out = reference(enc, enc_mask_indices, target_mask_indices)

    torch.testing.assert_close(optimized_out, reference_out, atol=1e-5, rtol=1e-5)


def test_precomputed_rope_positions_match_attention_helper():
    predictor = VisionTransformerPredictor(
        img_size=32,
        patch_size=16,
        num_frames=1,
        embed_dim=64,
        predictor_embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_rope=True,
    ).eval()
    masks = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int64)

    expected = predictor.predictor_blocks[0].attn.separate_positions(masks.unsqueeze(1))
    actual = _get_precomputed_rope_positions(predictor.predictor_blocks, masks)

    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(actual_part, expected_part)


def test_precomputed_rope_positions_match_attention_helper_non_square():
    predictor = VisionTransformerPredictor(
        img_size=(32, 64),
        patch_size=16,
        num_frames=1,
        embed_dim=96,
        predictor_embed_dim=96,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_rope=True,
    ).eval()
    masks = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 0, 1, 2, 3]], dtype=torch.int64)

    expected = predictor.predictor_blocks[0].attn.separate_positions(
        masks.unsqueeze(1), predictor.grid_height, predictor.grid_width
    )
    actual = _get_precomputed_rope_positions(predictor.predictor_blocks, masks, predictor.grid_height, predictor.grid_width)

    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(actual_part, expected_part)


def test_src_predictor_rope_has_cls_runs_cpu():
    predictor = VisionTransformerPredictor(
        img_size=32,
        patch_size=16,
        num_frames=1,
        embed_dim=96,
        predictor_embed_dim=96,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_rope=True,
    ).eval()

    B = 2
    enc_mask_indices = [torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)]
    target_mask_indices = [torch.tensor([[2, 3], [0, 3]], dtype=torch.int64)]
    enc = torch.randn(B, enc_mask_indices[0].size(1) + 1, 96)

    with torch.no_grad():
        out = predictor(enc, enc_mask_indices, target_mask_indices, has_cls=True)

    assert out.shape == (B, target_mask_indices[0].size(1), 96)


def test_sorted_target_positions_match_reverse_argsort():
    masks_x = torch.tensor([[1, 4, 6, 8], [0, 3, 5, 9]], dtype=torch.int64)
    masks_y = torch.tensor([[0, 2, 7], [1, 4, 8]], dtype=torch.int64)
    masks = torch.cat([masks_x, masks_y], dim=1)
    argsort = torch.argsort(masks, dim=1)
    reverse_argsort = torch.argsort(argsort, dim=1)

    expected = reverse_argsort[:, masks_x.size(1) :]
    actual = _get_sorted_target_positions(masks_x, masks_y)

    torch.testing.assert_close(actual, expected)
