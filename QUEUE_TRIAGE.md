# Queue Triage

## Status Counts
- `FAILED_PARITY`: 94
- `REJECTED`: 34
- `APPROVED`: 11
- `PARTIAL_WIN`: 8
- `FAILED_BENCHMARK`: 1

## Failure Categories
### `IMPORT_OR_SYNTAX_ERROR` (7)
- `fused_channel_shift`: 1
- `fused_chunked_softmax`: 1
- `fused_percentile_clip`: 1
- `fused_rms_residual`: 1
- `fused_subtract_mean_scale`: 1
- `fused_swiglu_block`: 1
- `queue/results/fused_channel_shift_001.json` -> `fused_channel_shift`
  - ==================================== ERRORS ==================================== ___________ ERROR collecting tests/queue/test_fused_channel_shift.py ___________ /usr/local/lib/python3.12/dist-packages/_pytest/python.py:
- `queue/results/fused_chunked_softmax_001.json` -> `fused_chunked_softmax`
  - ==================================== ERRORS ==================================== __________ ERROR collecting tests/queue/test_fused_chunked_softmax.py __________ /usr/local/lib/python3.12/dist-packages/_pytest/python.py:
- `queue/results/fused_percentile_clip_001.json` -> `fused_percentile_clip`
  - ==================================== ERRORS ==================================== __________ ERROR collecting tests/queue/test_fused_percentile_clip.py __________ /usr/local/lib/python3.12/dist-packages/_pytest/python.py:
- `queue/results/fused_rms_residual_001.json` -> `fused_rms_residual`
  - ==================================== ERRORS ==================================== ___________ ERROR collecting tests/queue/test_fused_rms_residual.py ____________ /usr/local/lib/python3.12/dist-packages/_pytest/python.py:
- `queue/results/fused_subtract_mean_scale_001.json` -> `fused_subtract_mean_scale`
  - ==================================== ERRORS ==================================== ________ ERROR collecting tests/queue/test_fused_subtract_mean_scale.py ________ /usr/local/lib/python3.12/dist-packages/_pytest/python.py:
- `queue/results/fused_swiglu_block_001.json` -> `fused_swiglu_block`
  - ==================================== ERRORS ==================================== ___________ ERROR collecting tests/queue/test_fused_swiglu_block.py ____________ /usr/local/lib/python3.12/dist-packages/_pytest/python.py:

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
- `queue/results/fused_add_norm_residual_002.json` -> `fused_add_norm_residual`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_norm_residual.py:22: in test_
- `queue/results/fused_add_norm_residual_004.json` -> `fused_add_norm_residual`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_norm_residual.py:22: in test_

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
- `queue/results/fused_index_select_mean_009.json` -> `fused_index_select_mean`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-small] _______________________ /usr/local/lib/python3.12/dist-packages/triton/language/
- `queue/results/fused_momentum_teacher_002.json` -> `fused_momentum_teacher`
  - F =================================== FAILURES =================================== ___________________ test_forward_parity[dtype0-vit_l_logits] ___________________ /usr/local/lib/python3.12/dist-packages/triton/language/

### `TRITON_COMPILE_ERROR` (16)
- `fused_add_bias`: 1
- `fused_add_relu`: 1
- `fused_add_silu`: 1
- `fused_add_tensors`: 1
- `fused_gather_add`: 1
- `fused_huber_loss`: 1
- `queue/results/fused_add_bias_001.json` -> `fused_add_bias`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_bias.py:16: in test_forward_p
- `queue/results/fused_add_relu_001.json` -> `fused_add_relu`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_relu.py:16: in test_forward_p
- `queue/results/fused_add_silu_001.json` -> `fused_add_silu`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_silu.py:16: in test_forward_p
- `queue/results/fused_add_tensors_001.json` -> `fused_add_tensors`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_tensors.py:16: in test_forwar
- `queue/results/fused_gather_add_002.json` -> `fused_gather_add`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-small] _______________________ /usr/local/lib/python3.12/dist-packages/triton/language/
- `queue/results/fused_huber_loss_001.json` -> `fused_huber_loss`
  - F =================================== FAILURES =================================== __________________________ test_parity[dtype0-vit_l] ___________________________ tests/queue/test_fused_huber_loss.py:16: in test_parity 

### `CUDA_RUNTIME_ERROR` (2)
- `fused_masked_fill_softmax`: 2
- `queue/results/fused_masked_fill_softmax_003.json` -> `fused_masked_fill_softmax`
  - F =================================== FAILURES =================================== _________________________ test_forward_parity[seq256] __________________________ /usr/local/lib/python3.12/dist-packages/torch/testing/_c
- `queue/results/fused_masked_fill_softmax_004.json` -> `fused_masked_fill_softmax`
  - F =================================== FAILURES =================================== _________________________ test_forward_parity[seq256] __________________________ /usr/local/lib/python3.12/dist-packages/torch/testing/_c

### `NUMERICAL_MISMATCH` (26)
- `fused_add_norm_residual`: 4
- `fused_squeeze_excitation`: 4
- `fused_gelu_linear`: 2
- `fused_l2_distance`: 2
- `fused_online_softmax`: 2
- `fused_token_scatter`: 2
- `queue/results/fused_abs_001.json` -> `fused_abs`
  - ......F =================================== FAILURES =================================== _________________________ test_backward_parity[vit_l] __________________________ tests/queue/test_fused_abs.py:30: in test_backward
- `queue/results/fused_add_bias_gelu_001.json` -> `fused_add_bias_gelu`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_bias_gelu.py:21: in test_forw
- `queue/results/fused_add_norm_residual_007.json` -> `fused_add_norm_residual`
  - .F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_h] _______________________ tests/queue/test_fused_add_norm_residual.py:24: in test
- `queue/results/fused_add_norm_residual_008.json` -> `fused_add_norm_residual`
  - .F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_h] _______________________ tests/queue/test_fused_add_norm_residual.py:24: in test
- `queue/results/fused_add_norm_residual_009.json` -> `fused_add_norm_residual`
  - ....F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype1-vit_h] _______________________ tests/queue/test_fused_add_norm_residual.py:24: in t
- `queue/results/fused_add_norm_residual_010.json` -> `fused_add_norm_residual`
  - .....F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype1-small] _______________________ tests/queue/test_fused_add_norm_residual.py:25: in 

### `BENCHMARK_ERROR` (1)
- `fused_drop_path_residual`: 1
- `queue/results/fused_drop_path_residual_001.json` -> `fused_drop_path_residual`
  - ......... [100%] 9 passed in 5.24s

### `BENCH_REGRESSION` (34)
- `fused_mish`: 3
- `fused_temporal_pool`: 3
- `fused_attn_transpose`: 2
- `fused_layernorm_add`: 2
- `fused_momentum_teacher`: 2
- `fused_norm_linear`: 2
- `queue/results/fused_argsort_gather_002.json` -> `fused_argsort_gather`
  - ......... [100%] 9 passed in 4.95s
- `queue/results/fused_attn_transpose_001.json` -> `fused_attn_transpose`
  - ......... [100%] 9 passed in 6.98s
- `queue/results/fused_attn_transpose_002.json` -> `fused_attn_transpose`
  - ......... [100%] 9 passed in 6.57s
- `queue/results/fused_cross_entropy_001.json` -> `fused_cross_entropy`
  - ... [100%] 3 passed in 5.61s
- `queue/results/fused_ema_update_001.json` -> `fused_ema_update`
  - ............ [100%] 12 passed in 6.40s
- `queue/results/fused_gather_add_003.json` -> `fused_gather_add`
  - ......... [100%] 9 passed in 6.94s

### `PARTIAL_BENCH_WIN` (8)
- `fused_add_norm_residual`: 1
- `fused_bilinear_act`: 1
- `fused_chunked_attention`: 1
- `fused_mish_gate`: 1
- `fused_sigmoid_mul`: 1
- `fused_silu_mul`: 1
- `queue/results/fused_add_norm_residual_001.json` -> `fused_add_norm_residual`
  - ............ [100%] 12 passed in 6.48s
- `queue/results/fused_bilinear_act_001.json` -> `fused_bilinear_act`
  - ......... [100%] 9 passed in 7.35s
- `queue/results/fused_chunked_attention_001.json` -> `fused_chunked_attention`
  - ... [100%] 3 passed in 3.29s
- `queue/results/fused_mish_gate_001.json` -> `fused_mish_gate`
  - ............ [100%] 12 passed in 8.31s
- `queue/results/fused_sigmoid_mul_001.json` -> `fused_sigmoid_mul`
  - ......... [100%] 9 passed in 7.81s
- `queue/results/fused_silu_mul_001.json` -> `fused_silu_mul`
  - ......... [100%] 9 passed in 6.46s

### `PARITY_ERROR` (19)
- `fused_rope_apply_fast`: 4
- `fused_gelu_linear`: 2
- `fused_mish`: 2
- `fused_qkv_split`: 2
- `fused_warp_reduce_sum`: 2
- `fused_add_tanh`: 1
- `queue/results/fused_add_tanh_001.json` -> `fused_add_tanh`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_add_tanh.py:16: in test_forward_p
- `queue/results/fused_clamp_001.json` -> `fused_clamp`
  - ......F =================================== FAILURES =================================== _________________________ test_backward_parity[vit_l] __________________________ tests/queue/test_fused_clamp.py:30: in test_backwa
- `queue/results/fused_dropout_residual_norm_001.json` -> `fused_dropout_residual_norm`
  - F =================================== FAILURES =================================== ____________________ test_forward_parity[dtype0-vit_small] _____________________ tests/queue/test_fused_dropout_residual_norm.py:23: in t
- `queue/results/fused_exp_001.json` -> `fused_exp`
  - ......F =================================== FAILURES =================================== _________________________ test_backward_parity[vit_l] __________________________ tests/queue/test_fused_exp.py:30: in test_backward
- `queue/results/fused_gelu_001.json` -> `fused_gelu`
  - F =================================== FAILURES =================================== ______________________ test_forward_parity[dtype0-vit_l] _______________________ tests/queue/test_fused_gelu.py:17: in test_forward_parit
- `queue/results/fused_gelu_linear_002.json` -> `fused_gelu_linear`
  - F =================================== FAILURES =================================== ____________________ test_forward_parity[dtype0-vit_l_proj] ____________________ tests/queue/test_fused_gelu_linear.py:19: in test_forwar

## Approved Kernels
- `queue/results/fused_3d_sincos_embed_005.json`
- `queue/results/fused_3d_sincos_embed_006.json`
- `queue/results/fused_adamw_step_001.json`
- `queue/results/fused_gradient_clip_001.json`
- `queue/results/fused_loss_001.json`
- `queue/results/fused_masked_fill_softmax_005.json`
- `queue/results/fused_rms_norm_001.json`
- `queue/results/fused_rope_apply_001.json`
- `queue/results/fused_scale_grad_001.json`
- `queue/results/fused_sincos_embed_002.json`
- `queue/results/fused_sincos_embed_003.json`
