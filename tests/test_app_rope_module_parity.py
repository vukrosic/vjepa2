# Copyright (c) Facebook, Inc. and affiliates.
#
# This source code is licensed under the license found in the
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

from app.vjepa_2_1.models.utils.modules import Block, RoPEAttention, rotate_query_key_pair


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


def _make_rope_kwargs():
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


def test_app_rotate_query_key_pair_matches_head_baseline_cuda():
    if not torch.cuda.is_available():
        return

    baseline = load_module_from_head(ROOT, "app/vjepa_2_1/models/utils/modules.py", "baseline_app_rope_pair")

    q = torch.randn(8, 16, 1029, 24, device="cuda", dtype=torch.float16)
    k = torch.randn(8, 16, 1029, 24, device="cuda", dtype=torch.float16)
    pos = torch.arange(1024, device="cuda", dtype=torch.float16)

    baseline_q = baseline.rotate_queries_or_keys(q, pos, n_registers=4, has_cls_first=True)
    baseline_k = baseline.rotate_queries_or_keys(k, pos, n_registers=4, has_cls_first=True)
    optimized_q, optimized_k = rotate_query_key_pair(q, k, pos, n_registers=4, has_cls_first=True)

    assert torch.equal(optimized_q, baseline_q)
    assert torch.equal(optimized_k, baseline_k)


def test_app_rope_attention_matches_head_baseline_cuda():
    if not torch.cuda.is_available():
        return

    baseline = load_module_from_head(ROOT, "app/vjepa_2_1/models/utils/modules.py", "baseline_app_rope_attention")
    kwargs = _make_rope_kwargs()

    optimized = RoPEAttention(**kwargs).cuda().half().eval()
    reference = baseline.RoPEAttention(**kwargs).cuda().half().eval()
    reference.load_state_dict(optimized.state_dict())

    B, T, H, W, C = 2, 4, 8, 8, 256
    x = torch.randn(B, 1 + (T * H * W) + 4, C, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        optimized_out, optimized_attn = optimized(x, T=T, H_patches=H, W_patches=W)
        reference_out, reference_attn = reference(x, T=T, H_patches=H, W_patches=W)

    assert optimized_attn is None and reference_attn is None
    assert torch.equal(optimized_out, reference_out)


def test_app_block_matches_head_baseline_cuda():
    if not torch.cuda.is_available():
        return

    baseline = load_module_from_head(ROOT, "app/vjepa_2_1/models/utils/modules.py", "baseline_app_block")
    kwargs = _make_rope_kwargs()
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
    reference = baseline.Block(**kwargs).cuda().half().eval()
    reference.load_state_dict(optimized.state_dict())

    B, T, H, W, C = 2, 4, 8, 8, 256
    x = torch.randn(B, 1 + (T * H * W) + 4, C, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        optimized_out, optimized_attn = optimized(x, T=T, H_patches=H, W_patches=W)
        reference_out, reference_attn = reference(x, T=T, H_patches=H, W_patches=W)

    assert optimized_attn is None and reference_attn is None
    assert torch.equal(optimized_out, reference_out)
