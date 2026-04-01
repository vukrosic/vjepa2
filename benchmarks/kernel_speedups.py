import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.utils.modules import ACRoPEAttention, RoPEAttention, rotate_queries_or_keys
from src.masks.utils import apply_masks


def baseline_rotate_queries_or_keys(x, pos):
    _, _, _, dim = x.size()
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freq.sin().squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = freq.cos().squeeze(-1).repeat(1, 1, 1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


def baseline_apply_masks(x, masks, concat=True):
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1)
        if m.ndim == 1:
            mask_keep = mask_keep.unsqueeze(0)
        mask_keep = mask_keep.expand(*mask_keep.shape[:-1], x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def baseline_rope_attention(module, x, mask=None, T=None, H_patches=None, W_patches=None):
    batch, tokens, channels = x.size()
    grid_depth = int(tokens // (module.grid_size * module.grid_size))
    qkv = module.qkv(x).unflatten(-1, (3, module.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    if mask is not None:
        mask = mask.unsqueeze(1).repeat(1, module.num_heads, 1)
        d_mask, h_mask, w_mask = module.separate_positions(mask, H_patches, W_patches)
    else:
        if T is None or H_patches is None or W_patches is None:
            mask = torch.arange(int(grid_depth * module.grid_size * module.grid_size), device=x.device)
        else:
            mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
        d_mask, h_mask, w_mask = module.separate_positions(mask, H_patches, W_patches)

    s = 0
    qd = baseline_rotate_queries_or_keys(q[..., s : s + module.d_dim], pos=d_mask)
    kd = baseline_rotate_queries_or_keys(k[..., s : s + module.d_dim], pos=d_mask)
    s += module.d_dim
    qh = baseline_rotate_queries_or_keys(q[..., s : s + module.h_dim], pos=h_mask)
    kh = baseline_rotate_queries_or_keys(k[..., s : s + module.h_dim], pos=h_mask)
    s += module.h_dim
    qw = baseline_rotate_queries_or_keys(q[..., s : s + module.w_dim], pos=w_mask)
    kw = baseline_rotate_queries_or_keys(k[..., s : s + module.w_dim], pos=w_mask)
    s += module.w_dim

    if s < module.head_dim:
        qr = q[..., s:]
        kr = k[..., s:]
        q = torch.cat([qd, qh, qw, qr], dim=-1)
        k = torch.cat([kd, kh, kw, kr], dim=-1)
    else:
        q = torch.cat([qd, qh, qw], dim=-1)
        k = torch.cat([kd, kh, kw], dim=-1)

    attn = (q @ k.transpose(-2, -1)) * module.scale
    attn = attn.softmax(dim=-1)
    x = attn @ v
    x = x.transpose(1, 2).reshape(batch, tokens, channels)
    x = module.proj(x)
    return module.proj_drop(x)


def baseline_ac_rope_attention(module, x, T, H, W, action_tokens):
    batch, tokens, channels = x.size()
    mask = torch.arange(int(T * H * W), device=x.device)
    d_mask, h_mask, w_mask = module.separate_positions(mask, H, W)
    h_mask = h_mask * (module.grid_size / H)
    w_mask = w_mask * (module.grid_size / W)

    if action_tokens > 0:
        x = x.view(batch, -1, action_tokens + H * W, channels)
        action_q, action_k, action_v = [], [], []
        for i in range(action_tokens):
            a = x[:, :, i : i + 1, :].flatten(1, 2)
            qkv = module.qkv(a).unflatten(-1, (3, module.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            depth_pos = torch.arange(T, device=x.device)
            qd = baseline_rotate_queries_or_keys(q[..., : module.d_dim], pos=depth_pos)
            kd = baseline_rotate_queries_or_keys(k[..., : module.d_dim], pos=depth_pos)
            qr = q[..., module.d_dim :]
            kr = k[..., module.d_dim :]
            action_q += [torch.cat([qd, qr], dim=-1).view(batch, module.num_heads, T, 1, -1)]
            action_k += [torch.cat([kd, kr], dim=-1).view(batch, module.num_heads, T, 1, -1)]
            action_v += [v.view(batch, module.num_heads, T, 1, -1)]

        action_q = torch.cat(action_q, dim=3).flatten(2, 3)
        action_k = torch.cat(action_k, dim=3).flatten(2, 3)
        action_v = torch.cat(action_v, dim=3).flatten(2, 3)
        x = x[:, :, action_tokens:, :].flatten(1, 2)

    qkv = module.qkv(x).unflatten(-1, (3, module.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    s = 0
    qd = baseline_rotate_queries_or_keys(q[..., s : s + module.d_dim], pos=d_mask)
    kd = baseline_rotate_queries_or_keys(k[..., s : s + module.d_dim], pos=d_mask)
    s += module.d_dim
    qh = baseline_rotate_queries_or_keys(q[..., s : s + module.h_dim], pos=h_mask)
    kh = baseline_rotate_queries_or_keys(k[..., s : s + module.h_dim], pos=h_mask)
    s += module.h_dim
    qw = baseline_rotate_queries_or_keys(q[..., s : s + module.w_dim], pos=w_mask)
    kw = baseline_rotate_queries_or_keys(k[..., s : s + module.w_dim], pos=w_mask)
    s += module.w_dim

    if s < module.head_dim:
        qr = q[..., s:]
        kr = k[..., s:]
        q = torch.cat([qd, qh, qw, qr], dim=-1)
        k = torch.cat([kd, kh, kw, kr], dim=-1)
    else:
        q = torch.cat([qd, qh, qw], dim=-1)
        k = torch.cat([kd, kh, kw], dim=-1)

    if action_tokens > 0:
        q = torch.cat([action_q.view(batch, module.num_heads, T, action_tokens, -1), q.view(batch, module.num_heads, T, H * W, -1)], dim=3).flatten(2, 3)
        k = torch.cat([action_k.view(batch, module.num_heads, T, action_tokens, -1), k.view(batch, module.num_heads, T, H * W, -1)], dim=3).flatten(2, 3)
        v = torch.cat([action_v.view(batch, module.num_heads, T, action_tokens, -1), v.view(batch, module.num_heads, T, H * W, -1)], dim=3).flatten(2, 3)

    attn = (q @ k.transpose(-2, -1)) * module.scale
    attn = attn.softmax(dim=-1)
    x = attn @ v
    x = x.transpose(1, 2).reshape(batch, tokens, channels)
    x = module.proj(x)
    return module.proj_drop(x)


def bench(name, baseline_fn, optimized_fn, warmup=20, iters=80):
    for _ in range(warmup):
        baseline_fn()
        optimized_fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        baseline_fn()
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - start) * 1000.0 / iters

    start = time.perf_counter()
    for _ in range(iters):
        optimized_fn()
    torch.cuda.synchronize()
    optimized_ms = (time.perf_counter() - start) * 1000.0 / iters

    with torch.no_grad():
        baseline_out = baseline_fn()
        optimized_out = optimized_fn()
        max_abs_diff = (baseline_out - optimized_out).abs().max().item()

    return {
        "name": name,
        "baseline_ms": baseline_ms,
        "optimized_ms": optimized_ms,
        "speedup_pct": ((baseline_ms - optimized_ms) / baseline_ms) * 100.0,
        "max_abs_diff": max_abs_diff,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    results = []

    apply_masks_x = torch.randn(8, 1024, 256, device="cuda", dtype=torch.float16)
    apply_masks_masks = [
        torch.randint(0, 1024, (8, 256), device="cuda"),
        torch.randint(0, 1024, (8, 256), device="cuda"),
        torch.randint(0, 1024, (8, 256), device="cuda"),
        torch.randint(0, 1024, (8, 256), device="cuda"),
    ]
    rotate_x = torch.randn(8, 16, 4096, 24, device="cuda", dtype=torch.float16)
    rotate_pos = torch.arange(4096, device="cuda").view(1, 1, -1)
    results.append(
        bench(
            "apply_masks_multi_2d",
            lambda: baseline_apply_masks(apply_masks_x, apply_masks_masks),
            lambda: apply_masks(apply_masks_x, apply_masks_masks),
            warmup=10,
            iters=50,
        )
    )

    results.append(
        bench(
            "rotate_queries_or_keys",
            lambda: baseline_rotate_queries_or_keys(rotate_x, rotate_pos),
            lambda: rotate_queries_or_keys(rotate_x, rotate_pos),
        )
    )

    rope = RoPEAttention(dim=1024, num_heads=16, use_sdpa=False, proj_drop=0.0, attn_drop=0.0, grid_size=16).cuda().eval()
    rope_x = torch.randn(2, 16 * 16 * 8, 1024, device="cuda")
    results.append(
        bench(
            "rope_attention_forward",
            lambda: baseline_rope_attention(rope, rope_x, T=8, H_patches=16, W_patches=16),
            lambda: rope(rope_x, T=8, H_patches=16, W_patches=16),
        )
    )

    ac_rope = ACRoPEAttention(dim=1024, num_heads=16, use_sdpa=False, proj_drop=0.0, attn_drop=0.0, grid_size=16).cuda().eval()
    ac_x = torch.randn(2, 8 * (3 + 16 * 16), 1024, device="cuda")
    results.append(
        bench(
            "ac_rope_attention_forward",
            lambda: baseline_ac_rope_attention(ac_rope, ac_x, T=8, H=16, W=16, action_tokens=3),
            lambda: ac_rope(ac_x, T=8, H=16, W=16, action_tokens=3),
            warmup=10,
            iters=40,
        )
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
