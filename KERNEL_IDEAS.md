# V-JEPA 2 Kernel Ideas

## Idea 1: fused_rmsn_residual
**Operation:** RMSNorm + residual add in a single pass. Computes the RMS scale factor, normalizes, scales by learned weight, then adds the skip connection all without writing intermediate results back to HBM.
**Source:** Any transformer block post-attention or post-MLP layer; analogous to the existing fused_layernorm_residual but without the mean subtraction step.
**Why it's fast:** Eliminates a full read+write of the activation tensor between RMSNorm and the residual add. RMSNorm is already cheaper than LayerNorm (no mean), so fusing the residual keeps the memory traffic at a single load + single store.
**Novelty:** RMSNorm is increasingly preferred over LayerNorm in large models for speed. A fused RMSNorm+residual kernel has not been widely published as a standalone Triton primitive; most implementations either do plain RMSNorm or fuse with something else.

## Idea 2: fused_rmsn_linear_residual
**Operation:** RMSNorm -> linear projection (GEMM) -> residual add, all fused into a single kernel that keeps the normalized tensor in shared memory/registers and feeds it directly into a tile-level GEMM epilogue before accumulating the residual.
**Source:** Post-attention projection path where RMSNorm precedes the output linear layer.
**Why it's fast:** The normalized activation tensor is produced and consumed within the same SRAM tile rather than being flushed to HBM between the norm and the GEMM. Saves two full tensor reads/writes for activations of shape [B, T, D].
**Novelty:** Combining a normalization pass with a GEMM is a compiler-level fusion that torch.compile rarely achieves cleanly due to shape constraints; a hand-written Triton kernel can tile them together explicitly.

## Idea 3: chunked_flash_attn_video
**Operation:** Flash-attention variant with an explicit temporal-chunk axis. Processes Q/K/V in blocks along the time dimension first, then the spatial dimension, exploiting the natural (time, height, width) token layout of video patches.
**Source:** Attention computation in the vision transformer; patches are laid out as [T, H, W] tubes before being flattened.
**Why it's fast:** Temporal chunks often attend sparsely to each other (nearby frames dominate). By tiling over the time axis first the kernel can skip entirely zero-masked blocks without loading them, reducing effective sequence length and HBM traffic proportionally.
**Novelty:** Standard FlashAttention tiles over the flat sequence dimension. A video-aware tiling strategy that respects the 3D patch lattice can exploit temporal locality and masking patterns specific to video prediction tasks.

## Idea 4: fused_cross_attn_kv_cache
**Operation:** Cross-attention between a query sequence (predictor tokens) and a key-value sequence (context/encoder tokens), fusing the KV projection + attention computation + output projection into a single kernel that keeps KV tiles in shared memory across multiple query blocks.
**Source:** Predictor module where context tokens from the target encoder serve as KV and predictor tokens serve as Q.
**Why it's fast:** KV tensors are read once from HBM and reused across all query tiles within a block. Without fusion the KV projection output must be materialized and then re-read for the attention computation.
**Novelty:** Cross-attention KV caching in a fused Triton kernel is uncommon; most frameworks materialize the projected KV. For V-JEPA 2's predictor, which attends encoder context repeatedly, this saves significant memory bandwidth.

## Idea 5: fused_rope_qk_split
**Operation:** Fused QKV linear projection -> split into Q, K, V buffers -> apply RoPE to Q and K, all in a single kernel pass. The RoPE frequencies are computed on-the-fly from precomputed cos/sin tables loaded once per block.
**Source:** Combined from existing fused_qkv_split and RoPE application; the split and RoPE are currently separate kernel launches.
**Why it's fast:** Eliminates an intermediate kernel launch and the associated HBM round-trip for Q and K tensors between QKV split and RoPE application.
**Novelty:** The existing fused_qkv_split stops at producing separate Q, K, V. Adding RoPE rotation inside the same kernel while reading Q, K only once is a natural extension that avoids a redundant memory pass.

## Idea 6: fused_3d_rope
**Operation:** 3D Rotary Position Embedding that independently encodes the temporal (t), height (h), and width (w) axes using separate frequency bands, applied to Q and K in a single fused kernel with no intermediate materialization.
**Source:** Position embedding logic; current RoPE likely uses a flattened 1D or 2D formulation for (h, w) with a separate time axis.
**Why it's fast:** A single kernel reads Q and K once, computes factored (t, h, w) rotation in registers, and writes rotated Q and K. Three separate 1D RoPE passes would require three reads of Q and K.
**Novelty:** 3D RoPE with factored axes is used in some video models but a single-pass fused Triton kernel with on-the-fly frequency computation for all three axes has not been widely published.

## Idea 7: fused_alibi_attn_bias
**Operation:** ALiBi (Attention with Linear Biases) position bias generation + addition to attention logits, fused into the attention kernel so the bias is computed on-the-fly from head index and token distance without materializing a full [B, H, T, T] bias matrix.
**Source:** Attention score computation; alternative to RoPE for positional encoding.
**Why it's fast:** A precomputed [B, H, T, T] ALiBi bias matrix for long video sequences is very large. Computing it on-the-fly inside the attention kernel tile means only the current tile's biases are ever in registers.
**Novelty:** ALiBi was designed for 1D sequences. A video-adapted ALiBi that uses 3D Manhattan distance (dt + dh + dw with per-axis slopes) fused directly into a Triton flash-attention kernel is a novel combination.

## Idea 8: fused_token_drop_compact
**Operation:** Token masking + compaction (gather) + layer norm, fused so that masked-out tokens are never written to HBM and the normalization runs over only the visible tokens. Outputs a compact dense tensor of visible tokens.
**Source:** Mask application before encoder forward pass; currently likely a separate mask-gather followed by layer norm.
**Why it's fast:** Avoids writing zero-padded masked tokens to HBM and immediately re-reading them. The compaction and normalization together require only a single read of the full token buffer and a single write of the compact buffer.
**Novelty:** Video models with high mask ratios (e.g., 90%) waste enormous bandwidth on padded tokens if masking and normalization are separate. Fusing them is particularly impactful for V-JEPA 2's masked prediction regime.

## Idea 9: fused_token_scatter_residual
**Operation:** Inverse of token compaction: scatter visible-token outputs back into a full-sequence buffer and add a residual (e.g., from a skip connection), writing zeros for masked positions.
**Source:** After encoder processes compact tokens and outputs need to be scattered back for loss computation or predictor input.
**Why it's fast:** Scatter and residual add share the same output address computation and single store per element. Separating them requires an intermediate full-buffer allocation and two passes.
**Novelty:** The inverse compaction with residual fusion is the natural pair to fused_token_drop_compact and is equally important for masked prediction architectures.

## Idea 10: fused_tubelet_embed_norm
**Operation:** 3D convolution (tubelet/patch embedding) + bias addition + layer norm + optional positional embedding addition, all fused. The conv output is kept in registers/shared memory and normalized before being written to HBM.
**Source:** Patch embedding module using a 3D conv with kernel (t_patch, h_patch, w_patch); first operation in the encoder.
**Why it's fast:** The tubelet embedding produces activations of shape [B, T, H, W, D] that are immediately normalized. Without fusion this requires a full write+read of a potentially large activation. With fusion, each output patch is normalized before being written.
**Novelty:** Fusing a 3D depthwise-separable style conv with LayerNorm in Triton is unusual; most fused embedding kernels only handle 2D spatial patches or linear projections.

## Idea 11: fused_swiglu_gate
**Operation:** SwiGLU with a learned gating scale: computes gate = SiLU(W1*x) * W2*x * sigmoid(g) where g is a per-channel learnable scalar, fused into a single activation kernel after the linear projections.
**Source:** MLP blocks using SwiGLU activation; extension of the existing fused_silu_mul.
**Why it's fast:** The learned gate scalar is broadcast across the batch and token dimensions; loading it once per channel inside a fused kernel avoids a separate elementwise multiply kernel launch.
**Novelty:** Adding a learnable gating scalar to SwiGLU is a simple architectural variant; the novelty is fusing it with SiLU and the elementwise multiply so the gating adds zero overhead over the base SwiGLU.

## Idea 12: fused_geglu
**Operation:** GeGLU activation: computes x_gate = GELU(W1*x), x_val = W2*x, output = x_gate * x_val. Fuses the GELU and the elementwise multiply, similar to fused_silu_mul but using GELU instead of SiLU.
**Source:** MLP blocks; GeGLU is an alternative to SwiGLU that some video transformers prefer.
**Why it's fast:** Same memory-traffic argument as fused_silu_mul: avoids writing the GELU output to HBM before multiplying elementwise.
**Novelty:** GeGLU has slightly different numerical characteristics from SwiGLU (exact GELU vs. SiLU approximation); a dedicated Triton kernel can use a fast GELU approximation (tanh-based) computed in registers, saving the separate GELU pass.

## Idea 13: fused_mish_gate
**Operation:** Mish activation (x * tanh(softplus(x))) fused with an elementwise gate multiply for a GLU variant: Mish-GLU.
**Source:** MLP activation; experimental activation function that sometimes outperforms SiLU on certain vision tasks.
**Why it's fast:** Mish requires computing softplus and tanh together; doing this in registers once and immediately gating avoids two separate activation passes.
**Novelty:** Mish-GLU as a fused Triton primitive for video transformers is unexplored. Mish's smooth, non-monotonic profile may provide better gradient flow for masked video prediction.

## Idea 14: fused_online_softmax_reduction
**Operation:** Online softmax (numerically stable) computed in a single pass using warp-level reductions, fused with the attention weight application to V, eliminating the two-pass softmax that standard implementations use.
**Source:** Self-attention score normalization; the core of FlashAttention's algorithm applied as a standalone Triton kernel for non-flash code paths.
**Why it's fast:** Standard softmax requires two passes (find max, then exp + sum). Online softmax with warp shuffles does it in one pass, halving memory traffic for large attention matrices.
**Novelty:** A standalone one-pass online softmax Triton kernel with warp-level reduce + update that can be composed with custom masking patterns specific to video (causal in time, non-causal in space) is novel.

## Idea 15: fused_causal_temporal_mask_attn
**Operation:** Attention with a causal mask along the temporal axis only (not the spatial axis), applied inside a flash-attention style kernel. Future time-step tokens are masked but all spatial tokens at the same or past timesteps attend freely.
**Source:** Temporal attention variant for autoregressive video generation or causal video prediction; encoder attention in causal video models.
**Why it's fast:** The causal temporal mask has a block structure when tokens are ordered as [T, H, W]. An attention kernel that exploits this block structure can skip future-time blocks without loading them.
**Novelty:** Most causal attention kernels apply causal masking to the flat sequence. A video-specific block-causal mask exploiting the [T, H, W] token ordering is a specialized optimization for video transformers.

## Idea 16: fused_grouped_query_attn
**Operation:** Grouped Query Attention (GQA) where K and V have fewer heads than Q, with the head-group broadcasting computed on-the-fly inside the attention kernel without materializing expanded K, V tensors.
**Source:** Attention module; GQA is a common efficiency technique for large transformers that V-JEPA 2 may adopt.
**Why it's fast:** Expanding K, V from G groups to H heads doubles memory traffic. Fusing the broadcast into the attention kernel means each K, V head group is loaded once and reused across the corresponding Q heads in registers.
**Novelty:** GQA-aware flash-attention in Triton with on-the-fly broadcast rather than pre-expansion is well-studied but a video-specific version that tiles over (T, H, W) rather than flat sequence is novel.

## Idea 17: fused_multi_query_attn_prefill
**Operation:** Multi-Query Attention (MQA, one K and V head shared across all Q heads) prefill kernel optimized for the large batch, long sequence regime of video training.
**Source:** Attention module; extreme case of GQA where G=1.
**Why it's fast:** With a single K, V head, the K and V tiles fit in L1 cache/shared memory and can be reused across all Q head tiles, dramatically reducing HBM bandwidth for K and V.
**Novelty:** MQA prefill for video transformers processes sequences of thousands of tokens (e.g., 8 frames x 14x14 spatial = 1568 tokens). A Triton kernel that tiles Q heads as the outer loop and keeps K, V in SRAM is especially effective at this scale.

## Idea 18: fused_local_temporal_attn
**Operation:** Local attention restricted to a temporal window of W frames. Each token only attends to tokens within the same spatial position but within [-W/2, +W/2] frames temporally. Implemented as a banded sparse attention kernel.
**Source:** Temporal self-attention layers in a hierarchical video transformer variant.
**Why it's fast:** Reduces attention complexity from O(T^2 * HW) to O(T * W * HW) for the temporal dimension. The banded structure allows the kernel to skip blocks outside the window entirely.
**Novelty:** A Triton kernel that combines banded temporal attention with dense spatial attention in the same operation, exploiting the (T, H, W) token layout for the band computation, is a natural video-specific optimization.

## Idea 19: fused_spatial_pool_attn
**Operation:** Spatial pooling of K and V (e.g., 2x2 average pool) before computing attention, fused so that pooling and the KV linear projection happen together without materializing the full-resolution KV tensors.
**Source:** Hierarchical or efficient attention layers that downsample spatial resolution for global context.
**Why it's fast:** The pooled KV tensors are smaller, reducing attention FLOPs quadratically in the spatial dimension. Fusing pooling with projection avoids writing the full-resolution projected KV to HBM.
**Novelty:** Pooled attention (as in Longformer or Perceiver) fused with KV projection in a single Triton kernel tuned for the (T, H, W) video token layout exploits spatial locality in a way generic attention kernels cannot.

## Idea 20: fused_xpos_rope
**Operation:** xPos (extrapolatable positional encoding) which applies position-dependent exponential scaling on top of RoPE, fused into a single Q/K rotation kernel.
**Source:** Positional encoding; xPos extends RoPE with better length generalization, useful for variable-length video sequences.
**Why it's fast:** xPos adds a per-position scalar scale on top of the RoPE rotation. Fusing the scale computation with the rotation avoids a separate elementwise multiply over [B, H, T, D/2] tensors.
**Novelty:** xPos for video (with 3D position indices) fused with the rotation in a single Triton kernel is unexplored; the temporal scaling could use a different decay rate than spatial scaling.

## Idea 21: fused_learnable_pe_add_norm
**Operation:** Learnable positional embedding lookup (index_select) + addition to patch embeddings + layer norm, fused so the embedding table is read once and added before normalization without materializing the intermediate sum.
**Source:** Positional embedding addition after patch embedding; common in ViT-based models.
**Why it's fast:** The positional embedding add and subsequent layer norm both touch the same tensor. Fusing them means the patch embedding + PE sum is computed in registers and normalized before being written to HBM.
**Novelty:** Index-select (gather) + add + norm as a single Triton kernel with the embedding table in shared memory is a useful primitive for variable-length video sequences where PE indices vary per sample.

## Idea 22: fused_sincos_3d_embed
**Operation:** 3D sincos positional embedding computation for (t, h, w) axes with configurable frequency bands per axis, outputting the summed embedding for each patch in a single pass without intermediate per-axis buffers.
**Source:** Extension of the existing fused_sincos_embed to full 3D; current implementation may flatten or use 2D+1D.
**Why it's fast:** Computing three sets of sincos embeddings and summing them requires three passes if done separately. Fusing into one kernel computes all frequencies in registers and accumulates the sum before writing.
**Novelty:** A Triton kernel that computes factored 3D sincos embeddings with configurable per-axis embedding dimensions (not necessarily equal thirds) and optional learned interpolation scales for temporal upsampling is novel for video models.

## Idea 23: fused_patch_embed_rope
**Operation:** Linear patch embedding (unfolded + matmul) + RoPE application, fused so that the embedded patch is rotated by its (h, w) position before being written to HBM.
**Source:** Patch embedding followed by positional encoding; alternative to additive PE that uses rotational encoding at the embedding stage.
**Why it's fast:** Applying RoPE immediately after projection means the embedded patch vector is in registers and can be rotated before the store, avoiding a separate RoPE kernel that re-reads the embeddings.
**Novelty:** Applying RoPE at the patch embedding stage rather than inside attention is an architectural variant (used in some efficient ViTs) that benefits from fusion when embedding and rotation share the same memory access.

## Idea 24: fused_fp16_layernorm_cast
**Operation:** LayerNorm computed in fp32 (for numerical stability) with the result cast to fp16 and written in a single kernel, accumulating in higher precision internally.
**Source:** Mixed-precision training pipeline; LayerNorm inputs are fp16 but the mean/variance computation needs fp32 accuracy.
**Why it's fast:** Avoids a separate cast kernel after LayerNorm. The fp32 accumulation happens in registers; only the final fp16 result is written to HBM.
**Novelty:** While many LayerNorm implementations cast internally, a Triton kernel that explicitly manages the fp32 accumulation lane and fp16 store with correct rounding mode and handles the residual add in the same pass is a carefully engineered primitive for mixed-precision video training.

## Idea 25: fused_bf16_rmsn_linear
**Operation:** RMSNorm in bf16 with fp32 accumulation for the scale factor, immediately followed by a linear projection where the normalized activations feed into a GEMM epilogue, all while keeping the activations on-chip between norm and projection.
**Source:** Any attention or MLP pre-norm + projection path in bf16 mixed-precision training.
**Why it's fast:** bf16 has 8 bits of mantissa; computing variance in bf16 introduces error. Accumulating in fp32 in registers costs no HBM bandwidth. The GEMM epilogue fusion avoids materializing normalized activations.
**Novelty:** A Triton kernel that does bf16 RMSNorm with fp32 variance accumulation and feeds directly into a bf16 GEMM is a precision-aware fusion not typically generated by torch.compile.

## Idea 26: fused_int8_quant_gemm_epilogue
**Operation:** Post-GEMM epilogue that dequantizes int8 accumulator results, applies activation (GELU or SiLU), and requantizes to int8 for the next layer, all in a single kernel without fp32 intermediate tensors written to HBM.
**Source:** Quantized inference path for MLP layers; int8 GEMM produces int32 accumulators that need conversion.
**Why it's fast:** The dequantize -> activate -> requantize chain is entirely elementwise and can run in the GEMM epilogue without an additional HBM round-trip. The bottleneck is compute, not memory.
**Novelty:** A Triton epilogue kernel for video transformer MLP blocks that handles per-channel dequantization scales, fused activation, and per-tensor output quantization is a practical step toward int8 video inference.

## Idea 27: fused_activation_quant
**Operation:** Activation quantization (compute scale = max(|x|) / 127, quantize to int8) fused with a preceding activation function (e.g., GELU) so that the post-activation value is quantized before being written to HBM.
**Source:** Any activation layer in the quantized inference path.
**Why it's fast:** The post-activation value is already in registers; computing the per-tensor or per-channel max and quantizing in the same kernel avoids a separate read of the activation tensor for quantization.
**Novelty:** Fusing activation function + quantization in a two-pass (max then quantize) single-kernel approach using shared memory for the inter-warp max reduction is more efficient than the typical separate-pass approach.

## Idea 28: fused_weight_quant_dequant
**Operation:** On-the-fly weight quantization (fp16 -> int8) and dequantization in the GEMM setup phase, fusing weight loading with quantization so that int8 weights are computed from fp16 masters on-the-fly during the matrix multiply.
**Source:** Quantization-aware training or GPTQ-style inference where weights are stored in fp16 but compute uses int8.
**Why it's fast:** Loading weights once and quantizing in registers before the GEMM avoids storing a separate int8 weight copy. For weight-memory-bound GEMMs, int8 weights reduce HBM bandwidth by 2x even when computed on-the-fly.
**Novelty:** On-the-fly weight quantization inside a Triton GEMM kernel (quantize-on-load) is an alternative to pre-quantized weights that works well for mixed-precision training where weights change each step.

## Idea 29: fused_grad_clip_adamw
**Operation:** Gradient clipping (global norm computation + scaling) fused with the AdamW optimizer step, so the clipped gradient is immediately used to update moments and parameters without being written to HBM as clipped_grad.
**Source:** Training loop; gradient clipping typically precedes the optimizer step but is implemented as a separate operation.
**Why it's fast:** The global norm requires a reduction over all parameters. Fusing it with AdamW means each parameter's gradient is read once, contributes to the global norm (via a two-pass approach with a pre-computed norm), and is immediately used for the parameter update.
**Novelty:** Two-kernel approach: first kernel computes global gradient norm; second fused kernel applies clipping + AdamW update. The second kernel reads each gradient only once rather than the current read (for clipping) + read (for AdamW) pattern.

## Idea 30: fused_loss_scale_backward
**Operation:** Loss scaling for mixed-precision training fused with the backward loss reduction: scales the loss by the current loss scale factor, computes the gradient, and checks for inf/nan in a single kernel, setting a flag that the optimizer reads.
**Source:** Mixed-precision training scaler; currently loss scaling and overflow checking are separate operations.
**Why it's fast:** The overflow check (scanning for inf/nan in gradients) is a reduction over all gradient values that is typically a separate kernel launch. Fusing it with the loss scaling step means gradients are scanned once as they are produced.
**Novelty:** A Triton kernel that simultaneously applies loss scaling to the output gradient and accumulates an inf/nan flag using warp-level ballot operations provides a minimal-latency overflow detection for fp16 training of video models.

## Idea 31: fused_ema_batchnorm_update
**Operation:** EMA update of a target encoder (existing fused_ema_update) extended to also update any batch-level statistics (running mean, running variance) for any BatchNorm layers in the target encoder, in a single kernel.
**Source:** Target encoder EMA update; if the model has BatchNorm layers in the target encoder, their running stats also need EMA-style updates.
**Why it's fast:** Combining parameter EMA and BatchNorm stats update into a single pass over the target encoder parameter tensors halves the number of kernel launches for the EMA step.
**Novelty:** Specific to joint EMA of weights + BatchNorm statistics in a unified kernel; relevant for hybrid architectures that combine self-attention with convolutional BatchNorm components.

## Idea 32: fused_dropout_residual_norm
**Operation:** Dropout -> residual add -> layer norm, three consecutive elementwise/reduction operations fused into a single kernel. The dropped activation is added to the residual in registers, then the sum is normalized before being written.
**Source:** Post-attention and post-MLP dropout + residual + norm pattern; extends fused_drop_path_residual by adding norm.
**Why it's fast:** Eliminates two intermediate HBM round-trips (after dropout and after residual add). All three operations are bandwidth-limited; fusing them collapses three passes into one.
**Novelty:** The three-way fusion of dropout + residual + norm is a more aggressive fusion than the existing fused_drop_path_residual; particularly impactful in deep models like ViT-H with 32 layers where this pattern repeats 64 times per forward pass.

## Idea 33: fused_warp_softmax
**Operation:** Softmax where each warp handles one row of the input matrix, using warp shuffle instructions to compute the max and sum reductions within a warp without shared memory, achieving lower latency than shared-memory-based softmax.
**Source:** Attention score softmax for small sequence lengths where shared memory overhead dominates.
**Why it's fast:** Warp shuffles have much lower latency than shared memory barriers. For sequence lengths <= 32 (fitting in one warp), this is optimal. For larger lengths, multiple warps cooperate using minimal shared memory for inter-warp communication.
**Novelty:** A Triton kernel that explicitly uses tl.reduce with warp-level semantics for the attention softmax, tuned for the token counts typical in video transformer attention heads (e.g., 196 spatial tokens per head per batch element).

## Idea 34: fused_prefix_sum_mask
**Operation:** Parallel prefix sum (scan) over a binary mask tensor to compute the offset of each valid (unmasked) token in the compact output, fused with the compaction gather, producing both the compact token buffer and the position-to-index mapping.
**Source:** Token masking and compaction for masked prediction; the prefix sum gives the scatter/gather indices needed for efficient compaction.
**Why it's fast:** A single-pass parallel prefix sum + gather in Triton avoids the two-pass (scan then gather) approach, reducing HBM traffic by materializing the offset array only briefly in shared memory.
**Novelty:** Fusing the prefix sum (to compute compact indices) with the actual gather (to produce compact tokens) in one Triton kernel is a stream-compaction primitive tailored for V-JEPA 2's high-mask-ratio video prediction.

## Idea 35: fused_block_sparse_attn
**Operation:** Block-sparse attention where the sparsity pattern follows the video patch structure: spatial tokens always attend to themselves and their spatial neighbors, but temporal attention is sparse (e.g., only keyframes). Implemented as a block-sparse Triton kernel.
**Source:** Attention module; video has natural block-sparse structure where neighboring spatial tokens are semantically related but distant temporal tokens may not be.
**Why it's fast:** Block sparsity maps directly to tile-level skipping in a Triton kernel. Blocks outside the sparsity pattern are never loaded. For a 50% sparse pattern, this halves attention FLOPs and memory traffic.
**Novelty:** A Triton block-sparse attention kernel that uses the (T, H, W) patch layout to define sparsity blocks (e.g., blocks are defined in T-H-W space rather than flattened sequence space) is a video-native sparse attention primitive.

## Idea 36: fused_register_token_attn
**Operation:** Attention where a small set of global "register" tokens attend to all patch tokens, combined with local attention among patch tokens, fused so that the register token attention and patch-patch local attention are computed in the same kernel.
**Source:** Register token mechanism (as in DINOv2 or similar models) where a few extra tokens aggregate global information.
**Why it's fast:** The register tokens' attention to all patches is dense but small (few register tokens). Fusing the dense register attention with the sparse patch-patch attention in one kernel avoids a separate full-attention kernel for registers.
**Novelty:** A dual-mode attention kernel that handles both sparse local patch attention and dense global register token attention in one pass, outputting updated register and patch tokens simultaneously, is a novel Triton kernel design.

## Idea 37: fused_3d_avg_pool_norm
**Operation:** 3D average pooling (temporal + spatial downsampling) followed by layer norm, fused so pooling outputs feed directly into normalization without HBM materialization.
**Source:** Hierarchical video transformer or stem of a convolution-attention hybrid architecture.
**Why it's fast:** Average pooling is memory-bandwidth-bound and produces a smaller tensor. Layer norm on the pooled tensor reads this smaller tensor once. Fusing means the pooled values are accumulated in shared memory and normalized before being written.
**Novelty:** 3D average pool + norm as a fused Triton kernel is useful for temporal downsampling stages in hierarchical video models, which V-JEPA 2 variants may adopt.

## Idea 38: fused_group_norm_act
**Operation:** GroupNorm + activation function (e.g., SiLU or GELU), fused so the normalized value is activated before being written to HBM.
**Source:** Convolutional components of the model (e.g., video patch embedding backbone); GroupNorm is preferred over BatchNorm for small batches common in video training.
**Why it's fast:** GroupNorm and the subsequent activation are both elementwise/reduction operations on the same tensor. Fusing eliminates the intermediate HBM round-trip between them.
**Novelty:** GroupNorm + SiLU (or GELU) as a Triton kernel with configurable group size, handling video tensors of shape [B, C, T, H, W] with contiguous or channels-last memory layouts, is practically useful for hybrid CNN-ViT video architectures.

## Idea 39: fused_instance_norm_residual
**Operation:** Instance normalization (per-sample, per-channel statistics) + residual add, fused for video feature maps of shape [B, C, T, H, W].
**Source:** Convolutional branches or style-transfer components; instance norm is used in some video generation and domain adaptation contexts.
**Why it's fast:** Instance norm statistics are computed per (B, C) slice; all spatial and temporal positions in the same (B, C) group can be reduced in a single kernel while simultaneously computing the residual add.
**Novelty:** A Triton instance norm + residual kernel for 5D video tensors with efficient reduction over the (T, H, W) axes using warp shuffles for the per-channel statistics is a video-specific normalization primitive.

## Idea 40: fused_attn_score_bias_rope
**Operation:** Attention score computation (QK^T / sqrt(d)) with simultaneous application of a learned relative position bias and RoPE in a single kernel, avoiding separate bias addition and rotation passes.
**Source:** Attention mechanism in models that combine RoPE with relative position biases (e.g., T5-style bias + RoPE).
**Why it's fast:** Adding a position bias to attention scores requires reading the score matrix once. If the score is computed and biased in the same kernel, the score tensor never needs to be fully materialized.
**Novelty:** Combining RoPE (multiplicative, applied to Q and K before dot product) with an additive relative position bias in a single attention kernel that computes QK^T while applying both position encodings is an unusual fusion.

## Idea 41: fused_lp_norm_loss_grad
**Operation:** Lp loss computation (forward) fused with its backward pass (gradient w.r.t. predictions), computing both the loss value and the gradient in a single kernel pass.
**Source:** Masked prediction loss; the existing fused_loss kernel likely only computes the forward loss value.
**Why it's fast:** The gradient of the Lp loss is elementwise (p * sign(diff) * |diff|^(p-1)) and can be computed immediately when the forward difference is available, without a separate backward pass that re-reads predictions and targets.
**Novelty:** A single-kernel forward+backward Lp loss is a custom autograd-style fusion that halves the number of passes over the prediction and target tensors, which are large for video models (many masked patches x embed_dim).

## Idea 42: fused_cosine_sim_loss
**Operation:** Cosine similarity loss between predictor output and target encoder output, fused with L2 normalization of both vectors in a single kernel. The norms are computed, both vectors normalized, and the dot product loss computed in one pass.
**Source:** Alternative or supplementary loss for masked video prediction; cosine similarity losses are common in self-supervised learning.
**Why it's fast:** L2 norm + normalize + dot product is three operations on the same two tensors. Fusing computes the norms in a first reduction pass, then normalizes and dots in a second pass, with both passes in the same kernel launch.
**Novelty:** A Triton kernel for cosine similarity loss that handles the two-reduction structure (per-vector norms, then dot product) using sequential warp reductions is a clean self-supervised learning primitive.

## Idea 43: fused_vicreg_loss
**Operation:** VICReg loss (variance + invariance + covariance terms) computed in a single kernel pass over a batch of embeddings, fusing the mean subtraction, covariance computation, and three loss terms.
**Source:** Alternative self-supervised loss for video representation learning; VICReg encourages embedding variance and decorrelation.
**Why it's fast:** The three VICReg terms all require the same mean-subtracted embeddings. Computing the mean once, then computing all three terms in a second pass (using shared accumulators) avoids three separate passes over the embedding batch.
**Novelty:** A fused Triton VICReg loss kernel for video embeddings of shape [B, T_masked, D] with online covariance estimation using Welford's algorithm is a specialized self-supervised learning kernel.

## Idea 44: fused_barlow_twins_cross_corr
**Operation:** Barlow Twins cross-correlation matrix computation + loss, fused so that the D x D correlation matrix is never fully materialized; instead the loss is accumulated as a sum of elementwise terms computed on-the-fly from the normalized embedding GEMM.
**Source:** Alternative self-supervised loss; Barlow Twins computes a cross-correlation matrix between two views' embeddings.
**Why it's fast:** The D x D matrix (D=1024 for ViT-L) is 4MB in fp32. Not materializing it and instead computing the loss terms as the GEMM epilogue saves the HBM write + read of this matrix.
**Novelty:** A Triton GEMM kernel with a custom epilogue that computes the Barlow Twins loss terms on-the-fly as matrix elements are computed, using warp-level reductions to accumulate the scalar loss, is a novel self-supervised learning kernel.

## Idea 45: fused_video_mae_mask_loss
**Operation:** Masked autoencoder loss (reconstruction loss over only masked patches) that combines mask application, Lp difference, and reduction in a single kernel, skipping visible (unmasked) patches entirely.
**Source:** Masked video prediction loss; the existing fused_loss may process all patches and mask out the loss values, wasting computation on visible patches.
**Why it's fast:** If visible patch losses are zeroed out after computation, FLOPs are wasted on those patches. A kernel that checks the mask before computing the difference skips ~10% of patches for a 90% mask ratio.
**Novelty:** A sparse loss kernel that uses a mask to select which patches to include in the loss computation, outputting the sum and count for normalization, with no wasted FLOPs on visible patches, is a natural optimization for high-mask-ratio video models.

## Idea 46: fused_attention_dropout_residual
**Operation:** Attention output (after softmax-weighted V sum) + dropout + residual add, fused so the attention output is dropped and residual-added before being written to HBM.
**Source:** Post-attention projection dropout + residual; currently a separate dropout and residual add after the attention output projection.
**Why it's fast:** Dropout generates a random mask per element and zeros some values; the residual add accumulates the result. Fusing means the attention output is read once, dropped, and added to the residual in a single store.
**Novelty:** Combining the final attention output with dropout and residual in a kernel that generates dropout masks using a Philox counter-based RNG (like Triton's tl.rand) avoids a separate random number generation pass.

## Idea 47: fused_weight_decay_update
**Operation:** L2 weight decay application fused directly into the AdamW step (which already does this conceptually), but with an additional fused step that computes the weight norm for monitoring purposes without extra memory traffic.
**Source:** Optimizer step; weight norm monitoring is useful for training stability diagnostics.
**Why it's fast:** Weight norm monitoring typically requires a separate reduction pass over parameters. Fusing it into the optimizer step means each parameter is accumulated into the norm sum while being updated, at no extra memory cost.
**Novelty:** A Triton AdamW kernel that simultaneously updates parameters and accumulates per-layer weight norms (for logging/debugging) is a practical training observability fusion.

## Idea 48: fused_gradient_centralization
**Operation:** Gradient centralization (subtracting the mean of gradients across output channels before the optimizer step) fused with the AdamW step, computing the gradient mean in registers and applying it before the moment update.
**Source:** Training optimization technique; gradient centralization often improves convergence of vision models.
**Why it's fast:** Gradient centralization requires computing the mean of the gradient tensor along specific axes and subtracting it. Fusing with AdamW means gradients are read once, centered, and used for moment updates without a separate centralization kernel.
**Novelty:** Gradient centralization + AdamW as a fused Triton kernel for 2D weight matrices (the common case in transformer linear layers) is a training efficiency optimization that is not part of standard PyTorch optimizers.

## Idea 49: fused_stochastic_depth_scale
**Operation:** Stochastic depth (DropPath) with per-layer drop rate scaling, where the survival probability scales linearly with depth, fused with the residual add so that the scaling factor is applied in the same kernel as the residual.
**Source:** Drop path residual; extends the existing fused_drop_path_residual with depth-dependent scaling without the scaling factor being a separate operation.
**Why it's fast:** The depth-dependent scaling is a scalar multiply that is trivially added to the residual add operation in registers.
**Novelty:** Making the DropPath survival probability a runtime parameter (rather than fixed at kernel compile time) allows the same kernel to serve all layers with depth-appropriate drop rates, reducing kernel specialization overhead.

## Idea 50: fused_layer_scale_residual
**Operation:** LayerScale (per-channel learnable scalar applied to the sublayer output) + residual add, fused into a single kernel. LayerScale is a broadcast multiply by a learnable vector of shape [D] followed by residual addition.
**Source:** Post-attention and post-MLP LayerScale in some ViT variants (e.g., CaiT-style); improves training stability for deep models.
**Why it's fast:** LayerScale is an elementwise broadcast multiply followed by an elementwise add. Fusing both into a single pass over the activation tensor (shape [B, T, D]) avoids two separate reads.
**Novelty:** LayerScale + residual as a Triton kernel where the scale vector is loaded once per output channel tile and broadcast across all batch and token positions is efficient for the [B, T, D] video activation shape.

## Idea 51: fused_qk_norm_attn
**Operation:** Q and K normalization (L2 or RMSNorm applied per head to Q and K vectors) fused with the QK^T attention score computation, so normalized Q and K are computed on-the-fly inside the attention kernel without writing normalized Q, K to HBM.
**Source:** QK normalization as used in some vision transformers to stabilize attention; the normalized Q and K are not needed outside the attention computation.
**Why it's fast:** Normalizing Q and K before attention requires reading them twice (once for norm, once for QK^T) if done separately. Fusing normalizes in registers and immediately computes the dot product.
**Novelty:** QK normalization fused into flash-attention-style tiled attention in Triton, where the per-head norm is computed as part of the Q/K tile loading phase, is a precision-stability optimization for deep video transformers.

## Idea 52: fused_rotary_freqs_cache
**Operation:** RoPE frequency computation (sin/cos tables) for arbitrary (t, h, w) positions, generating the cos/sin cache on-the-fly from frequency parameters rather than pre-materializing the entire [T_max, H_max, W_max, D] cache.
**Source:** RoPE setup; the frequency cache is typically pre-computed and stored, taking memory proportional to max sequence length.
**Why it's fast:** For variable-length video sequences (different frame counts at different training stages), a pre-materialized cache may be larger than needed or require recomputation. On-the-fly frequency generation uses only the memory needed for the current batch.
**Novelty:** A Triton kernel that computes RoPE frequencies on-demand from base frequency and position indices, eliminating the need for a stored frequency table and enabling arbitrary sequence length generalization.

## Idea 53: fused_head_norm_attn_output
**Operation:** Per-head L2 normalization applied to the attention output (value-weighted sum per head) before head concatenation, fused with the attention output computation.
**Source:** Some efficient attention variants that normalize per-head outputs before concatenation to control scale.
**Why it's fast:** Per-head normalization after attention would require a separate pass over the [B, H, T, D/H] attention output. Fusing it with the output computation normalizes each head's output while it is still in registers after the value-weighted sum.
**Novelty:** Head-wise normalization of attention outputs inside the attention kernel (rather than as a separate operation) is a novel stability technique for large video transformers that avoids a separate memory pass.

## Idea 54: fused_3d_depthwise_conv_norm
**Operation:** 3D depthwise convolution (spatial-temporal local mixing) + batch/layer norm, fused for video tensors of shape [B, C, T, H, W].
**Source:** Convolutional temporal mixing layers in hybrid CNN-ViT video architectures; depthwise convolution is used for local feature mixing before self-attention.
**Why it's fast:** Depthwise 3D conv has very low compute intensity (each output depends on only kernel_size^3 inputs); it is memory-bandwidth-bound. Fusing with norm eliminates the intermediate tensor store+load.
**Novelty:** A Triton kernel for 3D depthwise convolution with arbitrary padding + layer norm, handling the (B, C, T, H, W) layout, is a practical building block for efficient video backbone designs.

## Idea 55: fused_temporal_shift_embed
**Operation:** Temporal shift operation (shifting feature channels by one frame in the time dimension, used in TSM-style models) + patch embedding addition, fused so that the shifted feature is added to the current-frame embedding without materializing the shifted tensor.
**Source:** Temporal mixing mechanism; temporal shift is a zero-parameter temporal modeling technique.
**Why it's fast:** A naive temporal shift copies a portion of the feature map by one time step, producing an intermediate tensor. Fusing the shift with the embedding addition means the shifted values are added on-the-fly during the read phase.
**Novelty:** Temporal shift + embedding fusion in Triton is a lightweight temporal modeling primitive that avoids any intermediate tensor allocation, useful for efficient video encoders.

## Idea 56: fused_patch_merge_norm
**Operation:** Patch merging (2x2 spatial concatenation of tokens, as in Swin Transformer) + linear projection + layer norm, fused for downsampling in hierarchical video transformers.
**Source:** Hierarchical patch merging layers; concatenates [B, T, H//2, W//2, 4D] tokens and projects to [B, T, H//2, W//2, D].
**Why it's fast:** The concatenation + linear projection is a gather + GEMM. Fusing with norm means the projected result is never written to HBM between projection and normalization.
**Novelty:** A Triton kernel for video patch merging (which involves a 3D gather of 2x2 spatial neighbors from the token layout) + linear + norm as a unified operation, handling the transposed indexing that makes naive implementations slow.

## Idea 57: fused_all_reduce_norm
**Operation:** In a distributed setting, fused all-reduce (sum across data-parallel replicas) + layer normalization, so that as partial sums arrive from the ring all-reduce, the final normalization is computed immediately without a separate pass.
**Source:** Distributed training; currently all-reduce completes and then the next kernel normalizes.
**Why it's fast:** Waiting for all-reduce to complete and then launching a separate norm kernel adds launch latency. A kernel that computes normalization as part of the all-reduce reduce-scatter + all-gather pipeline can overlap computation with communication.
**Novelty:** Fusing the final all-reduce aggregation step with layer normalization in a custom Triton kernel that runs on the compute stream while all-reduce uses the communication stream is an advanced distributed training optimization.

## Idea 58: fused_reduce_scatter_norm_all_gather
**Operation:** Sequence parallelism primitive: reduce-scatter the activations across sequence parallel ranks, apply layer norm on the local shard, and all-gather the result, with the norm computation fused with the local shard processing.
**Source:** Sequence parallelism for very long video sequences (e.g., training on 64+ frames); splits the token sequence across GPUs.
**Why it's fast:** The layer norm in sequence parallelism operates on a local shard after reduce-scatter. Fusing the norm with the local reduce-scatter computation allows the norm to run as data arrives, overlapping with communication.
**Novelty:** A Triton kernel that does the local accumulation step of reduce-scatter and layer norm together, designed for the sequence-parallel training of video transformers where the sequence (token) dimension is distributed.

## Idea 59: fused_kv_compression_attn
**Operation:** Linear attention with KV compression: projects the key sequence to a smaller dimension using a learned compressor, then computes attention with compressed KV, fused so the compression and attention happen in the same kernel.
**Source:** Efficient attention for long video sequences; compressing KV reduces the O(T^2) attention to O(T * k) where k << T.
**Why it's fast:** The KV compression (a small GEMM) produces a small K, V tensor that fits in shared memory for the entire attention computation. Fusing means the compressed KV is computed on-the-fly in SRAM.
**Novelty:** Intra-kernel KV compression (the compression GEMM and the attention using its output run in the same kernel) is an unusual fusion that makes the compressed KV invisible to the memory subsystem.

## Idea 60: fused_nystrom_attn_landmark
**Operation:** Nyström-method approximated attention using landmark tokens: compute attention between queries and landmarks, between landmarks and keys (reusing landmarks as keys), and reconstruct the full attention, all in a single kernel.
**Source:** Efficient attention approximation; Nyström attention reduces O(T^2) to O(T * m) where m is the number of landmarks.
**Why it's fast:** The two small attention computations (Q to landmarks, V from landmarks) can both fit in shared memory for typical landmark counts (m=32-64). Fusing avoids materializing intermediate Q-landmark and landmark-K attention matrices.
**Novelty:** A Triton Nyström attention kernel that selects landmarks via strided sampling of the video token sequence (exploiting temporal regularity) and fuses both small attention computations is a video-specific efficient attention primitive.

## Idea 61: fused_performer_attn
**Operation:** FAVOR+ (Performers) random feature attention: applies random orthogonal feature maps to Q and K to approximate the softmax kernel, computes the linear attention in O(T*D) time, fused into a single kernel that generates features and computes the linear attention.
**Source:** Efficient attention approximation for very long video sequences (1000+ tokens per sample).
**Why it's fast:** Random feature computation (sinusoidal projections of Q, K) + matrix products for linear attention are all memory-bandwidth-bound. Fusing the feature computation with the linear attention accumulation avoids materializing the feature-mapped Q, K tensors.
**Novelty:** A Triton FAVOR+ kernel that uses on-the-fly random feature generation (computing the random projection matrix from a seed rather than storing it) and fuses with the two-step linear attention computation is a memory-efficient video attention primitive.

## Idea 62: fused_window_attn_shift
**Operation:** Shifted-window self-attention (Swin-style) where queries in each window attend only to keys in the same (possibly shifted) window, fused with the cyclic shift operation so that tokens are shifted and windowed in registers without materializing the shifted tensor.
**Source:** Window attention for spatial efficiency in video transformers; shifting ensures cross-window communication.
**Why it's fast:** The cyclic shift is typically implemented as a memory permutation (torch.roll). Fusing the shift with the window attention kernel reads tokens in the shifted order directly from HBM, skipping the intermediate shifted tensor.
**Novelty:** A Triton kernel that implements the cyclic shift by adjusting the HBM load addresses (computing shifted indices on-the-fly) and then performs window attention is an elegant way to avoid the torch.roll intermediate tensor.

## Idea 63: fused_video_window_3d_attn
**Operation:** 3D window attention for video: tokens within a (tw, hw, ww) spatial-temporal window attend to each other, with cyclic shifts in all three dimensions. Fused with window partition and inverse partition operations.
**Source:** Video Swin Transformer-style attention; the 3D window partitioning is specific to video and involves reshaping [B, T, H, W, D] -> [B*nT*nH*nW, tw*hw*ww, D].
**Why it's fast:** The reshape for window partitioning is a stride permutation that can be handled by adjusting memory access patterns in the kernel rather than materializing the rearranged tensor.
**Novelty:** A Triton 3D window attention kernel that uses strided loads to read tokens from the [B, T, H, W, D] layout into window-partitioned shared memory tiles, performing attention and writing back in the original layout, avoids costly reshape operations for video Swin-style transformers.

## Idea 64: fused_sparse_video_attn_topk
**Operation:** Top-k sparse attention for video: for each query token, attends only to the top-k key tokens by attention score, with the selection fused into the attention kernel using a streaming top-k selection algorithm.
**Source:** Efficient attention for long video sequences; dynamically selecting the most relevant tokens reduces attention cost for long temporal contexts.
**Why it's fast:** A streaming top-k kernel processes keys in tiles, maintaining a priority queue (heap) of size k in registers, computing attention only for selected keys. This avoids loading the full attention matrix.
**Novelty:** An online top-k attention kernel in Triton that maintains per-query heaps in registers while streaming over key tiles is a hardware-aware sparse attention primitive for video transformers.

## Idea 65: fused_hash_attn_lsh
**Operation:** Locality-Sensitive Hashing (LSH) attention: hash-based token grouping + attention within groups, fused so that the hashing (random projection + argmax) and group-wise attention run in the same kernel.
**Source:** Reformer-style efficient attention; LSH reduces attention to O(T log T) by attending within hash buckets.
**Why it's fast:** The hashing step (computing random projections of Q) shares the Q reads with the attention computation. Fusing hashing + bucket-wise attention avoids materializing the hash buckets as a separate tensor.
**Novelty:** A Triton LSH attention kernel for video that uses temporally-aware hash functions (giving higher probability of matching tokens in nearby frames) exploits video's temporal locality to improve LSH bucket quality.

## Idea 66: fused_warp_gather_matmul
**Operation:** Warp-cooperative gather (reading non-contiguous token embeddings using warp-level shuffles to distribute the work) immediately feeding into a matrix multiply, so scattered token embeddings are assembled and multiplied in a single pass.
**Source:** Any operation that requires selecting a subset of tokens (e.g., reading predictor input tokens from the encoder's output at specific masked positions) and immediately projecting them.
**Why it's fast:** A naive gather followed by GEMM requires writing the gathered tokens to HBM and re-reading them for the GEMM. Fusing keeps gathered values in registers/shared memory as they are immediately used as GEMM inputs.
**Novelty:** A Triton kernel that cooperatively gathers non-contiguous token rows from a large [N, D] embedding matrix using warp shuffles and feeds them directly into a tiled GEMM is a custom memory-access primitive for masked video transformers.

## Idea 67: fused_index_select_embed
**Operation:** Index-select (fancy indexing along the token dimension) + embedding addition + optional normalization, fused so that the selected tokens are fetched, have their positional embeddings added, and are optionally normalized in a single kernel.
**Source:** Any place where a subset of tokens is selected by index and needs positional re-embedding (e.g., when predictor tokens are at specific masked positions).
**Why it's fast:** Index-select produces a temporary tensor that is immediately consumed by embedding addition. Fusing eliminates the intermediate tensor entirely.
**Novelty:** A Triton index-select + embedding add kernel that reads from two separate embedding tables (patch embeddings and positional embeddings) and adds them together while writing only the final result is a clean IO-optimal gather primitive.

## Idea 68: fused_segment_sum_loss
**Operation:** Segmented reduction (sum or mean) over groups of tokens defined by a segment index array, used for computing per-video or per-tube losses, fused with the loss computation.
**Source:** Loss aggregation; when computing masked prediction loss per video clip or per temporal segment, a segmented reduction is needed.
**Why it's fast:** A segmented sum in Triton using shared memory accumulators indexed by segment avoids the overhead of sorting + scatter_add that PyTorch typically requires for this operation.
**Novelty:** A Triton segmented reduction kernel that accumulates per-segment losses in shared memory using atomic adds, without requiring pre-sorted segment indices, is a flexible loss aggregation primitive for video batch processing.

## Idea 69: fused_temperature_scaled_softmax
**Operation:** Temperature-scaled softmax (divides logits by a learnable temperature before softmax) fused with the attention score computation, with the temperature applied inline before the online softmax max-subtraction step.
**Source:** Contrastive loss softmax (e.g., CLIP-style or InfoNCE loss) or soft-label distillation in the V-JEPA 2 training setup.
**Why it's fast:** Temperature scaling is a scalar division applied to all logits; it commutes with the softmax max-subtraction. Applying it in the softmax kernel (as part of the division by sqrt(d_k)) avoids a separate elementwise operation.
**Novelty:** For video-language or video-video contrastive losses, a fused temperature-scaled softmax that handles large [B, B] similarity matrices with efficient warp-level online softmax is a practical training primitive.

## Idea 70: fused_cross_entropy_smooth
**Operation:** Label-smoothed cross-entropy loss computed in a single pass: softmax + log + label smoothing + scalar loss accumulation, without materializing the softmax probabilities or the log-softmax vector.
**Source:** Classification head or distillation loss; label smoothing is a common regularization technique.
**Why it's fast:** Cross-entropy with label smoothing is typically computed as log_softmax + weighted sum with smoothed labels. Fusing the log_softmax, label smoothing application, and loss accumulation into a single kernel avoids writing and reading the log-softmax vector.
**Novelty:** A Triton label-smoothed cross-entropy kernel that uses online log-sum-exp (single-pass numerically stable computation) for the log-partition function and accumulates the loss without materializing intermediate softmax probabilities is a clean training primitive.

## Idea 71: fused_mixup_embed
**Operation:** Mixup data augmentation (linear interpolation of two samples) applied to patch embeddings, fused with the embedding computation so that two samples' embeddings are blended in a single kernel rather than embedding each separately and then blending.
**Source:** Data augmentation in training; mixup can be applied at the embedding level rather than the pixel level for efficiency.
**Why it's fast:** Embedding two separate samples and then blending requires two embedding kernel launches + a blend kernel. Fusing reads both samples and produces their blended embedding in a single pass.
**Novelty:** A Triton mixup embedding kernel for video patches that blends two video patch embedding operations with a scalar lambda, producing a single mixed embedding, is a data-augmentation-aware memory optimization.

## Idea 72: fused_cutmix_mask_embed
**Operation:** CutMix augmentation at the patch level (replacing a cuboid of patches from one video with patches from another) fused with the patch embedding, so that the mix is determined by a spatial-temporal mask applied during embedding rather than after.
**Source:** Data augmentation; CutMix at the patch level is a natural extension for video (using a temporal-spatial cuboid mask).
**Why it's fast:** A naive CutMix embeds both videos and then copies patches based on the mask. A fused kernel selects the source sample (A or B) per patch during embedding, reading each patch's pixels only once.
**Novelty:** A Triton patch embedding kernel that reads from two different video input buffers, selecting per-patch source based on a 3D cuboid mask, is a data-augmentation-specific IO-optimal embedding primitive.

## Idea 73: fused_norm_quant_cast
**Operation:** Layer norm (in fp32) + quantization to int8 (or fp8) + type cast, all in a single kernel. The normalized value is quantized before being written, avoiding the fp32 intermediate in HBM.
**Source:** Quantized inference path; after normalization, activations need to be quantized for int8 GEMMs.
**Why it's fast:** Without fusion, LayerNorm writes fp32 or fp16 activations to HBM, and a separate quantization kernel reads and quantizes them. Fusing quantizes before the store, cutting HBM traffic from 4 bytes/element (fp32) to 1 byte/element (int8).
**Novelty:** A Triton LayerNorm + int8 quantization kernel that computes per-tensor or per-channel quantization scales using a warp-level max reduction and applies them within the same kernel as the normalization is a practical int8 inference primitive for video transformers.

## Idea 74: fused_fp8_cast_norm
**Operation:** fp16 -> fp8 (e4m3 or e5m2) cast + layer norm applied to fp8 inputs, where the norm accumulates in fp32 and outputs fp8, exploiting the Hopper H100 GPU's native fp8 support.
**Source:** fp8 training or inference; H100 GPUs support fp8 GEMM natively, and activations need to be in fp8 format.
**Why it's fast:** fp8 activations are half the size of fp16, so HBM bandwidth for activation tensors is halved. A norm kernel that reads fp8 and outputs fp8 (accumulating stats in fp32 internally) avoids format conversion overhead.
**Novelty:** A Triton LayerNorm kernel that operates natively in fp8 (using fp32 accumulators for statistics), exploiting Hopper-specific fp8 instructions for the element-wise operations, is a cutting-edge precision fusion for next-generation video model training.

## Idea 75: fused_sparse_moe_gate
**Operation:** Sparse mixture-of-experts (MoE) gating: compute expert scores (softmax over expert logits), select top-k experts per token, and compute routing weights, fused into a single kernel that outputs expert assignments and weights without materializing the full [T, num_experts] score matrix.
**Source:** MoE variants of V-JEPA 2 where some MLP layers are replaced by sparse MoE layers for parameter efficiency.
**Why it's fast:** The top-k selection over the expert score matrix can be done with a partial sort in registers. For small num_experts (8-32), the entire score vector fits in registers per token, making the full [T, num_experts] materialization unnecessary.
**Novelty:** A Triton MoE gating kernel for video transformers that handles the 3D token structure [B, T_sequence, D] with batch-level expert capacity constraints and outputs sparse routing indices is a video-aware MoE primitive.

## Idea 76: fused_expert_dispatch_compute
**Operation:** MoE expert dispatch (token-to-expert routing based on gating indices) + expert linear computation, fused so that tokens are routed and processed by their assigned expert without materializing a permuted token buffer.
**Source:** MoE forward pass; after gating, tokens are typically sorted by expert assignment, processed by each expert's linear layer, and un-sorted.
**Why it's fast:** The permutation sort + expert linear + un-permutation requires three memory passes. A fused dispatch kernel can process each token directly by its assigned expert, reading each token once and writing the result once.
**Novelty:** A Triton MoE dispatch kernel for video transformers that avoids explicit token permutation by directly addressing each expert's input tokens using scatter/gather indices during the GEMM is a hardware-efficient MoE primitive.

## Idea 77: fused_convnext_block
**Operation:** ConvNeXt block (depthwise conv 7x7 + LayerNorm + two linear layers with GELU + residual) fully fused: the depthwise conv output feeds into LayerNorm feeds into the two-layer MLP, all without intermediate HBM writes.
**Source:** ConvNeXt-style hybrid blocks in video backbones; ConvNeXt is increasingly used as a CNN component in hybrid video architectures.
**Why it's fast:** A ConvNeXt block has five operations; each currently requires a separate kernel. Fusing the depthwise conv + norm + GELU MLP eliminates four intermediate HBM round-trips.
**Novelty:** A full ConvNeXt block as a single Triton kernel (using a tiled strategy where each thread block handles a spatial region's full block computation) is an aggressive kernel fusion for hybrid video architectures.

## Idea 78: fused_rope_backward
**Operation:** Backward pass of RoPE rotation, computing gradients w.r.t. Q and K simultaneously in a single kernel (the backward of rotation is the inverse rotation applied to the output gradient).
**Source:** RoPE backward pass during training; the inverse rotation is another rotation by the negative angle.
**Why it's fast:** The RoPE backward gradient is identical in structure to the forward (apply the negative-angle rotation). A fused backward kernel computes grad_Q and grad_K together in a single pass over the grad_output tensor.
**Novelty:** A fused RoPE backward kernel that computes gradients for Q and K simultaneously (both are rotations of grad_output) without separate kernel launches is the natural complement to the existing RoPE forward kernel.

## Idea 79: fused_attn_backward_qkv
**Operation:** Flash-attention backward pass computing grad_Q, grad_K, grad_V simultaneously, using the saved attention output and log-sum-exp from the forward pass to recompute attention weights, all in a single tiled backward kernel.
**Source:** Attention backward pass; FlashAttention 2/3 already implements this, but a custom Triton version tuned for V-JEPA 2's specific head counts and sequence lengths may be more efficient.
**Why it's fast:** The flash-attention backward recomputes softmax from saved O and LSE rather than storing the attention matrix. Recomputing is cheaper than loading a [B, H, T, T] matrix from HBM.
**Novelty:** A V-JEPA 2-tuned flash-attention backward Triton kernel with specific tile sizes for the ViT-L (H=16, D_head=64) and ViT-H (H=16, D_head=80) configurations, exploiting the specific head dimension for warp layout optimization.

## Idea 80: fused_checkpoint_recompute
**Operation:** Gradient checkpointing-aware fused kernel that recomputes and immediately uses intermediate activations in the backward pass, combining the recomputation of a LayerNorm + activation function with the backward elementwise operations in a single kernel.
**Source:** Activation checkpointing in deep model training; V-JEPA 2 ViT-H with 32 layers requires checkpointing for memory efficiency.
**Why it's fast:** When recomputing checkpointed activations in the backward pass, the recomputed value is immediately used for gradient computation. Fusing recomputation + gradient computation avoids writing the recomputed activation to HBM.
**Novelty:** A Triton kernel that performs checkpointed forward recomputation (e.g., LayerNorm) and immediately computes the backward gradient w.r.t. the input in a single pass, outputting only the gradient (not the recomputed activation), is a memory-efficient backward kernel.

## Idea 81: fused_momentum_update_ema
**Operation:** Combined momentum buffer update (for SGD or LAMB optimizer) + EMA target encoder update, co-scheduled to reuse the same parameter reads for both the optimizer step and the EMA update.
**Source:** Training loop where both optimizer momentum and EMA target encoder need the current online encoder parameters.
**Why it's fast:** Both operations read the same online encoder parameters. Processing them together means each parameter tensor is read once from HBM and used for both the momentum update and the EMA update.
**Novelty:** A joint optimizer-momentum + target-EMA kernel that processes online encoder parameters in a single pass, producing updated momentum buffers and updated target parameters simultaneously, is unique to joint-embedding architectures like V-JEPA 2.

## Idea 82: fused_soft_dtw_loss
**Operation:** Soft Dynamic Time Warping (soft-DTW) loss for temporal sequence alignment between predictor and target encoder outputs, computed efficiently using a diagonal wavefront approach in a single Triton kernel.
**Source:** Alternative temporal alignment loss for video prediction; soft-DTW aligns predicted and target temporal sequences more flexibly than elementwise Lp loss.
**Why it's fast:** DTW requires a dynamic programming table of size [T1, T2]. The wavefront diagonal approach allows computing all cells in the same anti-diagonal in parallel, fitting each diagonal in shared memory.
**Novelty:** A Triton soft-DTW loss kernel for video embeddings that uses shared memory for the DP table and warp-level parallelism for diagonal elements is a novel temporal alignment primitive for video prediction models.

## Idea 83: fused_ssm_scan
**Operation:** State-space model (SSM / Mamba-style) selective scan operation, fused into a single parallel prefix scan kernel for sequence mixing as an alternative to attention in video transformers.
**Source:** Mamba/SSM layers as attention alternatives; SSMs process sequences in O(T) time with parallel scan during training.
**Why it's fast:** The SSM scan is a parallel prefix scan (Blelloch scan) that can be implemented efficiently in Triton with warp-level reductions and shared memory. A fused kernel computes the scan and applies the output gate in one pass.
**Novelty:** A Triton SSM scan kernel tuned for video sequence lengths (thousands of tokens per video clip) with chunk-wise parallelism and hardware-aware tile sizes for H100/A100 is a video-specific sequence mixing primitive.

## Idea 84: fused_retnet_recurrent_step
**Operation:** RetNet (Retentive Network) chunk-wise retention computation, fused into a kernel that processes the sequence in chunks using both the parallel and recurrent forms, switching adaptively.
**Source:** RetNet-style sequence mixing as an alternative to attention in video encoders; provides linear inference complexity.
**Why it's fast:** The chunk-wise retention combines intra-chunk parallel computation with inter-chunk recurrent state passing. Fusing both modes in a single Triton kernel avoids the overhead of separate parallel and recurrent kernels.
**Novelty:** A Triton RetNet kernel for video that uses larger chunks for spatial tokens (many tokens per frame) and smaller chunks for temporal tokens (using recurrent mode for efficient long-range temporal dependencies) exploits video structure.

## Idea 85: fused_linear_attn_norm
**Operation:** Linear attention (kernel-based, O(T) complexity) with feature normalization of Q and K, fused with the sequential numerator/denominator accumulation for the normalizer-free linear attention formulation.
**Source:** Efficient attention alternative; linear attention processes video's long sequences in O(T) time.
**Why it's fast:** Linear attention requires two sums: numerator (sum of k_i^T * v_i * q_j) and denominator (sum of k_i^T * q_j). Fusing these accumulations into a single scan over the sequence avoids two separate passes.
**Novelty:** A fused linear attention Triton kernel for video that accumulates both the KV outer product state and the K running sum in shared memory, processing the (T, H, W) sequence with temporal locality, is a video-native linear attention primitive.

## Idea 86: fused_masked_cosine_anneal
**Operation:** Learning rate cosine annealing + weight decay schedule update + masked parameter freezing (freezing EMA target encoder parameters from gradient updates), all in a single kernel that processes the parameter list.
**Source:** Training loop scheduling; currently separate Python-level operations that result in many kernel launches.
**Why it's fast:** Learning rate and weight decay are scalars applied to all parameter gradients. Combining their application with the optimizer step into a single kernel with runtime scalar arguments avoids recomputing schedules per parameter in separate kernels.
**Novelty:** A Triton optimizer kernel that takes cosine-annealed lr and weight decay as runtime scalars (computed by the scheduler) and applies them within the AdamW step, also checking a frozen_mask to skip certain parameter groups, is a clean training infrastructure fusion.

## Idea 87: fused_layer_drop_train
**Operation:** LayerDrop (randomly dropping entire transformer layers during training) implemented as a kernel that, for dropped layers, performs only the residual add (skipping the sublayer), with a compact conditional execution path.
**Source:** Training regularization; LayerDrop improves model robustness to depth variation.
**Why it's fast:** When a layer is dropped, only the residual pass-through is needed. A fused kernel that checks the drop flag per layer and either performs the full sublayer + residual or just the residual avoids a Python-level branch that causes separate kernel launches.
**Novelty:** A Triton kernel that conditionally executes LayerNorm + attention or just passes through the residual based on a per-layer drop mask (set per batch or per iteration), avoiding the overhead of Python-level conditional branching for layer drops.

## Idea 88: fused_tubelet_position_index
**Operation:** Computation of (t, h, w) position indices for each patch in a video batch, given variable-length video clips (different numbers of frames per sample in the batch), fused with positional embedding lookup.
**Source:** Variable-length video batching; different samples in a batch may have different temporal lengths, requiring per-sample position index computation.
**Why it's fast:** Positional index computation for variable-length batches involves branch-heavy index arithmetic. A Triton kernel that computes indices and immediately performs the embedding lookup avoids the Python-level index computation overhead.
**Novelty:** A Triton kernel that handles ragged batches of video (variable frame counts) by computing cumulative patch offsets and using them for correct positional embedding lookup is a practical primitive for variable-length video training.

## Idea 89: fused_sequence_pack_norm
**Operation:** Sequence packing (concatenating multiple variable-length video samples into a single flat sequence for efficient attention) + layer normalization applied per-sample in the packed sequence.
**Source:** Efficient batching; packing avoids padding overhead for variable-length video batches.
**Why it's fast:** Sequence packing is a memory rearrangement; normalization that follows must know per-sample boundaries. Fusing both means the packing (gather of per-sample tokens) feeds directly into per-sample normalization in the same kernel.
**Novelty:** A Triton sequence packing + normalization kernel that uses per-sample statistics (not pooled across the packed batch) by using document boundary indices to reset running accumulators is a novel variable-length video training primitive.

## Idea 90: fused_temporal_interpolate_embed
**Operation:** Temporal positional embedding interpolation for variable frame-rate inputs: given pre-computed sincos embeddings at a reference frame rate, interpolate them for the current frame timestamps using linear or sinusoidal interpolation, fused with the embedding addition.
**Source:** Training on videos with variable frame rates; the positional embedding needs to reflect actual timestamps, not just frame indices.
**Why it's fast:** Interpolation is a two-point weighted sum that can be computed in registers. Fusing with the embedding addition means the interpolated embedding is added to patch embeddings without being materialized.
**Novelty:** A Triton temporal positional embedding interpolation kernel that takes continuous timestamps (not discrete frame indices) and computes the interpolated embedding on-the-fly is a novel primitive for V-JEPA 2's variable-frame-rate training scenario.

## Idea 91: fused_contrastive_gather_loss
**Operation:** Contrastive loss with in-batch negative gathering across distributed ranks: gathers embeddings from other GPUs, computes pairwise similarities, and accumulates the loss in a single kernel that treats the inter-GPU gather as the outer loop.
**Source:** Distributed contrastive learning (e.g., VideoNCE or CLIP-style loss) for video-language or video-video training.
**Why it's fast:** Gathering negatives from other GPUs creates large embedding matrices. A kernel that processes one GPU's worth of embeddings at a time, accumulating partial loss sums without materializing the full inter-GPU embedding matrix, saves memory.
**Novelty:** A Triton distributed contrastive loss kernel that processes sharded negatives in tiles (as they arrive via all-gather) and accumulates partial loss values before the all-reduce is a practical distributed training primitive.

## Idea 92: fused_spectral_norm_compute
**Operation:** Spectral normalization's power iteration (one or more steps to estimate the largest singular value of a weight matrix) fused with the weight normalization application, in a single kernel.
**Source:** Discriminator or stable training components; spectral normalization is used to constrain Lipschitz constants in video generation models.
**Why it's fast:** Power iteration (u -> Wv / |Wv|, v -> W^T u / |W^T u|) requires two matrix-vector products. Fusing these with the subsequent weight normalization (dividing W by sigma) avoids materializing the normalized weight separately.
**Novelty:** A Triton spectral normalization kernel that performs one power iteration step and applies the normalization in a single kernel, updating the u and v vectors in-place and outputting the normalized weight tile, is a self-contained Lipschitz-constraint primitive.

## Idea 93: fused_relative_position_bias
**Operation:** Relative position bias for 3D video attention (separate learned biases for temporal and spatial relative positions), generated on-the-fly from a compact bias table and added to attention logits within the attention kernel.
**Source:** Attention with 3D relative position biases (as in Video Swin or Video BEiT); the bias table has shape [2*T-1, 2*H-1, 2*W-1] and is indexed by relative position.
**Why it's fast:** Materializing the full [T^2, H^2, W^2] attention bias from the compact table is expensive. Computing the bias index and looking up the table on-the-fly inside the attention kernel, with the table in shared memory, avoids materializing the full bias matrix.
**Novelty:** A Triton flash-attention kernel with an inner loop that computes 3D relative position indices (dt, dh, dw) and looks up biases from a shared-memory table is a video-native relative position bias fusion that avoids the O(T^2 H^2 W^2) memory cost.

## Idea 94: fused_video_mae_decoder_embed
**Operation:** Masked autoencoder decoder embedding: for each masked position, look up a learnable mask token embedding, add the position embedding for that position, and pass through the decoder, all fused so that mask token + position embedding is computed in a single kernel per masked patch.
**Source:** V-JEPA 2 predictor/decoder; masked positions need mask token embeddings with correct positional embeddings.
**Why it's fast:** Currently: broadcast mask token -> add position embedding -> (optional) norm. Fusing combines the broadcast (which is just a load of the mask token for each masked position) with position embedding addition.
**Novelty:** A Triton kernel that generates masked token embeddings for all masked positions simultaneously, using a compact mask token vector (broadcast) + per-position sincos embedding (computed on-the-fly), outputting only the masked-position embeddings in compact form.

## Idea 95: fused_hierarchical_loss_weight
**Operation:** Hierarchical loss weighting for multi-scale video prediction: computes loss at multiple temporal scales (frame-level, clip-level, video-level) with scale-dependent weights, fusing the multi-scale reduction and weighting in a single kernel.
**Source:** Multi-scale masked prediction loss; predicting at multiple temporal granularities requires losses at each scale.
**Why it's fast:** Multiple separate loss reductions (one per temporal scale) each require full passes over the prediction tensor. A single kernel that accumulates scale-wise losses in parallel shared memory buckets reduces HBM traffic to a single pass.
**Novelty:** A Triton hierarchical loss kernel that simultaneously reduces predictions over multiple temporal granularities (e.g., per-frame, per-2-frames, per-clip) using shared memory accumulators for each scale is a video-specific multi-scale loss primitive.

## Idea 96: fused_two_stream_cross_attn
**Operation:** Two-stream cross-attention where an online encoder stream (query) attends to a target encoder stream (key/value), with the linear projections for Q (from online) and KV (from target) fused together and the cross-attention kernel following immediately.
**Source:** Joint-embedding architecture where the predictor cross-attends to both its own tokens and the target encoder's tokens; two separate projection kernels are currently used.
**Why it's fast:** The Q projection (from online features) and KV projections (from target features) can be computed in parallel if the two feature buffers are both in HBM; a fused kernel reads both simultaneously and produces Q, K, V in a single memory pass.
**Novelty:** A Triton kernel that reads from two separate feature buffers (online and target encoder outputs) in a single kernel launch, computing Q from one and KV from the other, is a joint-embedding-architecture-specific fusion with no analog in single-encoder transformers.

## Idea 97: fused_sinkhorn_normalizer
**Operation:** Sinkhorn-Knopp normalization (doubly stochastic normalization of a cost/score matrix) for optimal transport-based assignment losses, computed iteratively with each Sinkhorn step fused with the row/column normalization in a single kernel per step.
**Source:** Optimal transport-based self-supervised losses (e.g., SwAV-style prototype assignment) applied to video representations.
**Why it's fast:** Each Sinkhorn step is a row softmax followed by a column normalization, which are two separate softmax-like operations. A Triton kernel that performs both in a single pass (using a two-phase approach within shared memory) halves the HBM traffic per Sinkhorn step.
**Novelty:** A Triton Sinkhorn normalization kernel for video prototype assignment, handling the large [B*T, num_prototypes] score matrix efficiently with tiled row and column normalization, is a novel self-supervised video learning primitive.

## Idea 98: fused_online_covariance_norm
**Operation:** Online (streaming) covariance computation of a batch of embeddings using Welford's algorithm, fused with feature normalization (whitening), in a single kernel that processes the batch once for covariance estimation and once for whitening application.
**Source:** Self-supervised learning objectives (e.g., VICReg covariance term, or whitening for W-MSE loss) applied to video embeddings.
**Why it's fast:** Welford's online covariance uses a single pass over the data. Fusing the covariance computation with the whitening application (which uses the computed covariance) in back-to-back kernels, with the covariance matrix in L2 cache between them, avoids HBM round-trips for the covariance matrix.
**Novelty:** A Triton Welford online covariance kernel that processes video embedding batches of shape [B, T_masked, D] and outputs whitened embeddings using the online-estimated covariance matrix is a video-specific self-supervised learning primitive.

## Idea 99: fused_attention_sink
**Operation:** Attention with "attention sinks" (initial tokens that receive disproportionate attention for stability, as in StreamingLLM) fused with the standard attention computation, adding a small number of sink tokens that are always kept in the KV cache and always attended to.
**Source:** Long video sequence processing; attention sinks stabilize attention for very long temporal contexts (many frames).
**Why it's fast:** Sink tokens are always included in the KV set regardless of windowing. A fused kernel that always loads sink token KV tiles first and reuses them across all query tiles avoids re-loading sink tokens for each query block.
**Novelty:** An attention sink Triton kernel for video transformers that preloads a small number of temporal anchor tokens (e.g., the first frame's tokens) into shared memory and keeps them as sinks while processing the rest of the video with sliding window attention is a video-specific memory management optimization.

## Idea 100: fused_time_warp_embed
**Operation:** Time-warping positional embedding for slow/fast temporal pathways: given a slow pathway (sparse frames) and a fast pathway (dense frames), computes positional embeddings for both temporal resolutions simultaneously in a single kernel, and generates the cross-pathway attention mask in the same pass.
**Source:** SlowFast-style dual-pathway video encoders where V-JEPA 2 may process slow and fast temporal streams; cross-pathway attention requires knowing the relative temporal positions of tokens from each pathway.
**Why it's fast:** Generating positional embeddings for two temporal resolutions separately requires two kernel launches. A joint kernel computes both sets of embeddings and their pairwise temporal distance matrix (for the cross-attention mask) in one pass over the temporal position arrays.
**Novelty:** A Triton kernel that simultaneously generates sincos positional embeddings for two temporal sampling rates and computes a binary attention mask based on temporal overlap between slow and fast frames is a novel dual-pathway video transformer primitive with no equivalent in image transformers.
