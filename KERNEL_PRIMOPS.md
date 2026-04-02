# Kernel Primops

This file defines the small set of kernel families that are safe to generate repeatedly in this repo.
The point is to keep the queue focused on patterns with a realistic chance of passing parity and winning benchmarks.

## Safe Families

### 1. Contiguous Elementwise
Use for:
- `add`, `sub`, `mul`, `div`
- `relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`, `square`, `abs`, `neg`
- scalar scale or bias variants

Rules:
- Only use when tensors are contiguous and the operation is truly bandwidth-bound.
- Prefer this only if it removes at least one full read/write pass.
- Do not expect a win for tiny tensors or for operations already fused by PyTorch.

### 2. Binary Broadcast
Use for:
- `x + bias`
- `x * scale`
- `x + residual`
- `activation(x) * gate`

Rules:
- Keep broadcast semantics simple and explicit.
- Validate rank and broadcast dimensions in `can_use_kernel()`.
- Do not mix complicated indexing with broadcast math in the same first-pass kernel.

### 3. Row-Wise Reduction
Use for:
- row sum
- row mean
- row RMS / variance
- Lp loss reductions
- softmax max/sum reduction

Rules:
- Reduce over one contiguous axis first.
- Initialize accumulators before the reduction loop.
- Use `tl.arange(0, BLOCK)` only with `tl.constexpr` block sizes.
- Keep the reduction axis obvious in both baseline and kernel.

### 4. Norm + Residual
Use for:
- LayerNorm + residual
- RMSNorm + residual
- dropout + residual + norm, only when the shapes are large enough to benefit

Rules:
- Compute statistics in fp32 even if inputs are fp16/bf16.
- Return exact backward grads or fall back to baseline.
- Do not fuse if the path is tiny enough that launch overhead dominates.

### 5. Linear Epilogue
Use for:
- GELU + linear
- SiLU + multiply + linear
- add/bias + linear
- projection + residual

Rules:
- Only attempt when the normalized or activated tensor can stay on-chip until the linear epilogue.
- Keep the baseline exact.
- If the implementation requires a full custom GEMM, do not fake it with unsafe indexing.

### 6. Attention Layout Helpers
Use for:
- QKV split and reshape
- attn transpose + reshape
- RoPE application on contiguous Q/K blocks
- masked softmax on stable block shapes

Rules:
- Use explicit shape assertions.
- Make sure head, sequence, and embedding axes are unambiguous.
- Keep index tensors integer typed.
- Prefer separate kernels over one giant speculative attention fusion.

### 7. Gather / Scatter
Use for:
- token gather
- token scatter
- index select + add

Rules:
- Verify index ranges and dtypes.
- Use scalar loads for scalar indices, block loads for block data.
- Never feed a block mask to a scalar pointer.
- Prefer gather/scatter only when the memory layout is already simple.

### 8. Positional Encoding
Use for:
- 1D sincos embed
- 3D sincos embed
- RoPE variants

Rules:
- Check divisibility constraints in baseline and kernel.
- Make the output shape match the test exactly.
- Do not assume flattened dimensions unless the test and source do.

## When Not To Fuse

Do not fuse when any of the following are true:

- The op is tiny and already memory cheap.
- The benchmark shape is small enough that Triton launch overhead dominates.
- The fusion would require a custom backward that is not already known to be correct.
- The fusion would need complicated atomics, cross-program communication, or dynamic control flow.
- The kernel would depend on opaque shape math or hidden layout assumptions.
- The current failure mode is already numerical instability rather than bandwidth waste.
- The fusion would duplicate work that the compiler or eager PyTorch already does well.

## Triton Safety Rules

- Use `tl.arange` only with `tl.constexpr` sizes.
- Use block masks only with block pointers.
- Do not call `tl.load` or `tl.store` with incompatible pointer and mask types.
- Initialize accumulators before entering loops.
- Use explicit `num_warps` and conservative `BLOCK` sizes first.
- Keep pointer arithmetic linear and easy to audit.
- If a kernel uses `autograd.Function`, save only valid tensors and metadata.
- If a kernel cannot run safely for a shape or dtype, fall back to baseline.

## Good Queue Targets

Prioritize these patterns:
- `fused_add_norm_residual`
- `fused_rms_residual`
- `fused_gelu_linear`
- `fused_silu_mul`
- `fused_sigmoid_mul`
- `fused_qkv_split`
- `fused_rope_apply`
- `fused_online_softmax`
- `fused_token_scatter`
- `fused_sincos_embed`

Avoid spending time on tiny elementwise kernels unless they are part of a bigger fusion that demonstrably reduces memory traffic.

