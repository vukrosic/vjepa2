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

from src.models.attentive_pooler import AttentivePooler
from src.models.utils.modules import ACRoPEAttention, RoPEAttention


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


def test_ac_rope_attention_matches_head_baseline_cpu():
    baseline = load_module_from_head(ROOT, "src/models/utils/modules.py", "baseline_src_modules_test_ac")

    kwargs = dict(dim=64, num_heads=4, qkv_bias=True, proj_drop=0.0, attn_drop=0.0, use_sdpa=False, grid_size=4)
    optimized = ACRoPEAttention(**kwargs).eval()
    reference = baseline.ACRoPEAttention(**kwargs).eval()
    reference.load_state_dict(optimized.state_dict())

    B, T, H, W, A, C = 2, 2, 4, 4, 2, 64
    x = torch.randn(B, T * (A + H * W), C)

    with torch.no_grad():
        optimized_out = optimized(x, T=T, H=H, W=W, action_tokens=A)
        reference_out = reference(x, T=T, H=H, W=W, action_tokens=A)

    torch.testing.assert_close(optimized_out, reference_out, atol=1e-5, rtol=1e-5)


def test_rope_attention_matches_head_baseline_cpu():
    baseline = load_module_from_head(ROOT, "src/models/utils/modules.py", "baseline_src_modules_test_rope")

    kwargs = dict(dim=64, num_heads=4, qkv_bias=True, proj_drop=0.0, attn_drop=0.0, use_sdpa=False, grid_size=4)
    optimized = RoPEAttention(**kwargs).eval()
    reference = baseline.RoPEAttention(**kwargs).eval()
    reference.load_state_dict(optimized.state_dict())

    B, T, H, W, C = 2, 2, 4, 4, 64
    x = torch.randn(B, T * H * W, C)

    with torch.no_grad():
        optimized_out = optimized(x, T=T, H_patches=H, W_patches=W)
        reference_out = reference(x, T=T, H_patches=H, W_patches=W)

    torch.testing.assert_close(optimized_out, reference_out, atol=1e-5, rtol=1e-5)


def test_attentive_pooler_matches_head_baseline_cpu():
    baseline = load_module_from_head(ROOT, "src/models/attentive_pooler.py", "baseline_src_attentive_pooler")

    kwargs = dict(
        num_queries=3,
        embed_dim=64,
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
    )
    optimized = AttentivePooler(**kwargs).eval()
    reference = baseline.AttentivePooler(**kwargs).eval()
    reference.load_state_dict(optimized.state_dict())

    x = torch.randn(2, 32, 64)

    with torch.no_grad():
        optimized_out = optimized(x)
        reference_out = reference(x)

    torch.testing.assert_close(optimized_out, reference_out, atol=1e-5, rtol=1e-5)
