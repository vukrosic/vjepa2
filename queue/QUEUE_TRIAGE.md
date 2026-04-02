# Queue Triage

Human report only. Kernel agents should ignore this file and follow `KERNEL_AGENT_WORKLIST.md`.

## Status Counts
- `BENCHMARK_REGRESSION`: 54
- `PARITY_NUMERICAL_MISMATCH`: 40
- `PARITY_TRITON_COMPILE_ERROR`: 35
- `PARITY_FAILURE`: 27
- `PARITY_BASELINE_OR_TEST_BUG`: 13
- `BENCHMARK_PARTIAL_WIN`: 13
- `BENCHMARK_WIN`: 11
- `PARITY_TRITON_POINTER_ERROR`: 11
- `PARITY_IMPORT_OR_SYNTAX_ERROR`: 9
- `PARITY_CUDA_RUNTIME_ERROR`: 2
- `BENCHMARK_ERROR`: 1

## Failure Categories
### `IMPORT_OR_SYNTAX_ERROR` (9)
- `fused_channel_shift`: 1
- `fused_chunked_softmax`: 1
- `fused_max_tensor`: 1
- `fused_min_tensor`: 1
- `fused_percentile_clip`: 1
- `fused_rms_residual`: 1
- `queue/results/fused_channel_shift_001.json` -> `fused_channel_shift`
  - ==================================== ERRORS ==================================== ___________ ERROR collecting tests/queue/test_fused_channel_shift.py ___________ /usr/local/lib/python3.12/dist-packages/_pytest/python.py:
- `queue/results/fused_chunked_softmax_001.json` -> `fused_chunked_softmax`
  - ==================================== ERRORS ==================================== __________ ERROR collecting tests/queue/test_fused_chunked_softmax.py __________ /usr/local/lib/python3.12/dist-packages/_pytest/python.py:
- `queue/results/fused_max_tensor_001.json` -> `fused_max_tensor`
  - ==================================== ERRORS ==================================== ____________ ERROR collecting tests/queue/test_fused_max_tensor.py _____________ ImportError while importing test module '/workspace/vjepa2
- `queue/results/fused_min_tensor_001.json` -> `fused_min_tensor`
  - ==================================== ERRORS ==================================== ____________ ERROR collecting tests/queue/test_fused_min_tensor.py _____________ ImportError while importing test module '/workspace/vjepa2

### `BASELINE_OR_TEST_BUG` (13)
- `fused_3d_sincos_embed`: 4
- `fused_add_norm_residual`: 2
- `fused_mlp_block`: 2
- `fused_gather_add`: 1
- `fused_geglu`: 1
- `fused_masked_softmax_rope`: 1
- `queue/results/fused_3d_sincos_embed_001.json` -> `fused_3d_sincos_embed`
  - F =================================== FAILURES =================================== ________________________ test_forward_parity[vit_l_16f] ________________________ tests/queue/test_fused_3d_sincos_embed.py:11: in test_fo
- `queue/results/fused_3d_sincos_embed_002.json` -> `fused_3d_sincos_embed`
  - F =================================== FAILURES =================================== ________________________ test_forward_parity[vit_l_16f] ________________________ tests/queue/test_fused_3d_sincos_embed.py:11: in test_fo
- `queue/results/fused_3d_sincos_embed_003.json` -> `fused_3d_sincos_embed`
  - F =================================== FAILURES =================================== ________________________ test_forward_parity[vit_l_16f] ________________________ tests/queue/test_fused_3d_sincos_embed.py:11: in test_fo
- `queue/results/fused_3d_sincos_embed_004.json` -> `fused_3d_sincos_embed`
  - F =================================== FAILURES =================================== ________________________ test_forward_parity[vit_l_16f] ________________________ tests/queue/test_fused_3d_sincos_embed.py:11: in test_fo

### `TRITON_POINTER_ERROR` (11)
- `fused_momentum_teacher`: 3
- `fused_softmax_cross_entropy_logits`: 3
- `fused_argsort_gather`: 2
- `fused_index_select_mean`: 2
- `fused_add_norm_residual`: 1
- `queue/results/fused_add_norm_residual_006.json` -> `fused_add_norm_residual`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ /usr/local/lib/python3.12/dist-packages/triton/language/
- `queue/results/fused_argsort_gather_001.json` -> `fused_argsort_gather`
  - ......F =================================== FAILURES =================================== _________________________ test_backward_parity[small] __________________________ /usr/local/lib/python3.12/dist-packages/triton/lan
- `queue/results/fused_argsort_gather_003.json` -> `fused_argsort_gather`
  - ......F =================================== FAILURES =================================== _________________________ test_backward_parity[small] __________________________ /usr/local/lib/python3.12/dist-packages/triton/lan
- `queue/results/fused_index_select_mean_002.json` -> `fused_index_select_mean`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-small] _______________________ /usr/local/lib/python3.12/dist-packages/triton/language/

### `TRITON_COMPILE_ERROR` (35)
- `fused_add_bias`: 2
- `fused_add_relu`: 2
- `fused_add_silu`: 2
- `fused_add_tensors`: 2
- `fused_div_tensors`: 2
- `fused_huber_loss`: 2
- `queue/results/fused_add_bias_001.json` -> `fused_add_bias`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_bias.py:16: in test_forward_p
- `queue/results/fused_add_bias_003.json` -> `fused_add_bias`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_bias.py:16: in test_forward_p
- `queue/results/fused_add_relu_001.json` -> `fused_add_relu`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_relu.py:16: in test_forward_p
- `queue/results/fused_add_relu_003.json` -> `fused_add_relu`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_relu.py:16: in test_forward_p

### `CUDA_RUNTIME_ERROR` (2)
- `fused_masked_fill_softmax`: 2
- `queue/results/fused_masked_fill_softmax_003.json` -> `fused_masked_fill_softmax`
  - F =================================== FAILURES =================================== _________________________ test_forward_parity[seq256] __________________________ /usr/local/lib/python3.12/dist-packages/torch/testing/_c
- `queue/results/fused_masked_fill_softmax_004.json` -> `fused_masked_fill_softmax`
  - F =================================== FAILURES =================================== _________________________ test_forward_parity[seq256] __________________________ /usr/local/lib/python3.12/dist-packages/torch/testing/_c

### `NUMERICAL_MISMATCH` (40)
- `fused_add_norm_residual`: 4
- `fused_squeeze_excitation`: 4
- `fused_gelu_linear`: 2
- `fused_l2_distance`: 2
- `fused_online_softmax`: 2
- `fused_token_scatter`: 2
- `queue/results/fused_abs_001.json` -> `fused_abs`
  - ......F =================================== FAILURES =================================== _________________________ test_backward_parity[vit_l] __________________________ tests/queue/test_fused_abs.py:30: in test_backward
- `queue/results/fused_abs_add_001.json` -> `fused_abs_add`
  - ......F =================================== FAILURES =================================== _________________________ test_backward_parity[vit_l] __________________________ tests/queue/test_fused_abs_add.py:37: in test_back
- `queue/results/fused_add_add_tensors_001.json` -> `fused_add_add_tensors`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_add_tensors.py:21: in test_fo
- `queue/results/fused_add_bias_gelu_001.json` -> `fused_add_bias_gelu`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_bias_gelu.py:21: in test_forw

### `BENCHMARK_ERROR` (1)
- `fused_drop_path_residual`: 1
- `queue/results/fused_drop_path_residual_001.json` -> `fused_drop_path_residual`
  - ......... [100%] 9 passed in 5.24s

### `BENCH_REGRESSION` (54)
- `fused_mish`: 3
- `fused_momentum_teacher`: 3
- `fused_temporal_pool`: 3
- `fused_argsort_gather`: 2
- `fused_attn_transpose`: 2
- `fused_celu`: 2
- `queue/results/fused_argsort_gather_002.json` -> `fused_argsort_gather`
  - ......... [100%] 9 passed in 4.95s
- `queue/results/fused_argsort_gather_004.json` -> `fused_argsort_gather`
  - .......... [100%] 10 passed in 3.72s
- `queue/results/fused_attn_transpose_001.json` -> `fused_attn_transpose`
  - ......... [100%] 9 passed in 6.98s
- `queue/results/fused_attn_transpose_002.json` -> `fused_attn_transpose`
  - ......... [100%] 9 passed in 6.57s

### `PARTIAL_BENCH_WIN` (13)
- `fused_3d_sincos_embed`: 1
- `fused_add_norm_residual`: 1
- `fused_add_scale`: 1
- `fused_bilinear_act`: 1
- `fused_chunked_attention`: 1
- `fused_mish_gate`: 1
- `queue/results/fused_3d_sincos_embed_007.json` -> `fused_3d_sincos_embed`
  - ... [100%] 3 passed in 3.20s
- `queue/results/fused_add_norm_residual_001.json` -> `fused_add_norm_residual`
  - ............ [100%] 12 passed in 6.48s
- `queue/results/fused_add_scale_001.json` -> `fused_add_scale`
  - ......... [100%] 9 passed in 7.16s
- `queue/results/fused_bilinear_act_001.json` -> `fused_bilinear_act`
  - ......... [100%] 9 passed in 7.35s

### `PARITY_ERROR` (27)
- `fused_rope_apply_fast`: 4
- `fused_gelu_linear`: 2
- `fused_hardtanh`: 2
- `fused_mish`: 2
- `fused_qkv_split`: 2
- `fused_quick_gelu`: 2
- `queue/results/fused_add_tanh_001.json` -> `fused_add_tanh`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_tanh.py:16: in test_forward_p
- `queue/results/fused_clamp_001.json` -> `fused_clamp`
  - ......F =================================== FAILURES =================================== _________________________ test_backward_parity[vit_l] __________________________ tests/queue/test_fused_clamp.py:30: in test_backwa
- `queue/results/fused_div_scale_001.json` -> `fused_div_scale`
  - ......F =================================== FAILURES =================================== _________________________ test_backward_parity[vit_l] __________________________ tests/queue/test_fused_div_scale.py:38: in test_ba
- `queue/results/fused_dropout_residual_norm_001.json` -> `fused_dropout_residual_norm`
  - F =================================== FAILURES =================================== ____________________ test_forward_parity[dtype0-vit_small] _____________________ tests/queue/test_fused_dropout_residual_norm.py:23: in t

## Approved Kernels
