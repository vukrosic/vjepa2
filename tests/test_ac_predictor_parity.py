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


def test_ac_predictor_matches_head_baseline_cpu():
    baseline = load_module_from_head(ROOT, "src/models/ac_predictor.py", "baseline_src_ac_predictor")

    kwargs = dict(
        img_size=32,
        patch_size=16,
        num_frames=4,
        tubelet_size=1,
        embed_dim=64,
        predictor_embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        use_rope=True,
        action_embed_dim=7,
    )
    optimized = VisionTransformerPredictorAC(**kwargs).eval()
    reference = baseline.VisionTransformerPredictorAC(**kwargs).eval()
    reference.load_state_dict(optimized.state_dict())

    B = 2
    T = kwargs["num_frames"] // kwargs["tubelet_size"]
    tokens = T * ((kwargs["img_size"] // kwargs["patch_size"]) ** 2)
    x = torch.randn(B, tokens, kwargs["embed_dim"])
    actions = torch.randn(B, T, kwargs["action_embed_dim"])
    states = torch.randn(B, T, kwargs["action_embed_dim"])

    with torch.no_grad():
        optimized_out = optimized(x, actions, states)
        reference_out = reference(x, actions, states)

    torch.testing.assert_close(optimized_out, reference_out, atol=1e-5, rtol=1e-5)


def test_ac_predictor_matches_head_baseline_cuda():
    if not torch.cuda.is_available():
        return

    baseline = load_module_from_head(ROOT, "src/models/ac_predictor.py", "baseline_src_ac_predictor_cuda")

    kwargs = dict(
        img_size=64,
        patch_size=16,
        num_frames=4,
        tubelet_size=1,
        embed_dim=128,
        predictor_embed_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        use_rope=True,
        action_embed_dim=7,
    )
    optimized = VisionTransformerPredictorAC(**kwargs).cuda().half().eval()
    reference = baseline.VisionTransformerPredictorAC(**kwargs).cuda().half().eval()
    reference.load_state_dict(optimized.state_dict())

    B = 2
    T = kwargs["num_frames"] // kwargs["tubelet_size"]
    tokens = T * ((kwargs["img_size"] // kwargs["patch_size"]) ** 2)
    x = torch.randn(B, tokens, kwargs["embed_dim"], device="cuda", dtype=torch.float16)
    actions = torch.randn(B, T, kwargs["action_embed_dim"], device="cuda", dtype=torch.float16)
    states = torch.randn(B, T, kwargs["action_embed_dim"], device="cuda", dtype=torch.float16)

    with torch.no_grad():
        optimized_out = optimized(x, actions, states)
        reference_out = reference(x, actions, states)

    torch.testing.assert_close(optimized_out, reference_out, atol=3e-3, rtol=3e-3)


def test_ac_predictor_matches_head_baseline_cuda():
    if not torch.cuda.is_available():
        return

    baseline = load_module_from_head(ROOT, "src/models/ac_predictor.py", "baseline_src_ac_predictor_cuda")

    kwargs = dict(
        img_size=64,
        patch_size=16,
        num_frames=8,
        tubelet_size=1,
        embed_dim=96,
        predictor_embed_dim=96,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        use_rope=True,
        action_embed_dim=7,
    )
    optimized = VisionTransformerPredictorAC(**kwargs).cuda().eval()
    reference = baseline.VisionTransformerPredictorAC(**kwargs).cuda().eval()
    reference.load_state_dict(optimized.state_dict())

    B = 2
    T = kwargs["num_frames"] // kwargs["tubelet_size"]
    tokens = T * ((kwargs["img_size"] // kwargs["patch_size"]) ** 2)
    x = torch.randn(B, tokens, kwargs["embed_dim"], device="cuda")
    actions = torch.randn(B, T, kwargs["action_embed_dim"], device="cuda")
    states = torch.randn(B, T, kwargs["action_embed_dim"], device="cuda")

    assert optimized.attn_mask.is_cuda
    assert reference.attn_mask.device.type == "cpu"

    with torch.no_grad():
        optimized_out = optimized(x, actions, states)
        reference_out = reference(x, actions, states)

    torch.testing.assert_close(optimized_out, reference_out, atol=1e-5, rtol=1e-5)


def test_ac_predictor_matches_head_baseline_cuda_extrinsics():
    if not torch.cuda.is_available():
        return

    baseline = load_module_from_head(ROOT, "src/models/ac_predictor.py", "baseline_src_ac_predictor_cuda_extrinsics")

    kwargs = dict(
        img_size=64,
        patch_size=16,
        num_frames=4,
        tubelet_size=1,
        embed_dim=96,
        predictor_embed_dim=96,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        use_rope=True,
        action_embed_dim=7,
        use_extrinsics=True,
    )
    optimized = VisionTransformerPredictorAC(**kwargs).cuda().eval()
    reference = baseline.VisionTransformerPredictorAC(**kwargs).cuda().eval()
    reference.load_state_dict(optimized.state_dict())

    B = 2
    T = kwargs["num_frames"] // kwargs["tubelet_size"]
    tokens = T * ((kwargs["img_size"] // kwargs["patch_size"]) ** 2)
    x = torch.randn(B, tokens, kwargs["embed_dim"], device="cuda")
    actions = torch.randn(B, T, kwargs["action_embed_dim"], device="cuda")
    states = torch.randn(B, T, kwargs["action_embed_dim"], device="cuda")
    extrinsics = torch.randn(B, T, kwargs["action_embed_dim"] - 1, device="cuda")

    assert optimized.attn_mask.is_cuda
    assert reference.attn_mask.device.type == "cpu"

    with torch.no_grad():
        optimized_out = optimized(x, actions, states, extrinsics=extrinsics)
        reference_out = reference(x, actions, states, extrinsics=extrinsics)

    torch.testing.assert_close(optimized_out, reference_out, atol=1e-5, rtol=1e-5)
