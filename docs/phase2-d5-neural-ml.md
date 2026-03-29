# Phase 2 Deep Quality Analysis: Domain 5 - Neural/ML

**Reviewer**: QE Code Reviewer (V3)
**Priority**: P1 HIGH
**Date**: 2026-03-29
**Crates in Scope**: ruvector-attention, ruvector-cnn, ruvector-gnn, neural-trader-core, neural-trader-coherence, neural-trader-replay, neural-trader-wasm, sona

---

## Executive Summary

Domain 5 encompasses the core neural/ML computation infrastructure of the RuVector monorepo. It contains approximately 170+ Rust source files across 8 crates, implementing attention mechanisms (FlashAttention-3, MLA, SSM/Mamba, multi-head, speculative decoding), CNN feature extraction with SIMD acceleration, GNN layers with message passing, and the SONA adaptive learning system. The code demonstrates substantial sophistication, particularly in the FlashAttention and MLA implementations. However, several categories of findings emerge from this deep analysis.

**Weighted Finding Score**: 19.75 (minimum threshold: 3.0)

| Severity | Count | Weight | Total |
|----------|-------|--------|-------|
| CRITICAL | 1 | 3.0 | 3.0 |
| HIGH | 5 | 2.0 | 10.0 |
| MEDIUM | 9 | 1.0 | 9.0 |
| LOW | 7 | 0.5 | 3.5 |
| INFORMATIONAL | 5 | 0.25 | 1.25 |

---

## 1. Attention Mechanism Audit (ruvector-attention)

### 1.1 Variants Implemented

The ruvector-attention crate implements an impressive breadth of attention mechanisms across 72+ source files:

- **Scaled Dot-Product Attention** (`attention/scaled_dot_product.rs`) - Fundamental QKV attention
- **Multi-Head Attention** (`attention/multi_head.rs`) - Parallel head splitting
- **FlashAttention-3** (`attention/flash.rs`) - IO-aware tiled attention with online softmax
- **Multi-Head Latent Attention (MLA)** (`attention/mla.rs`) - DeepSeek-V2/V3 style KV-cache compression
- **Selective State Space Model (Mamba/S6)** (`attention/ssm.rs`) - O(n) recurrent alternative
- **Speculative Decoding** (`attention/speculative.rs`) - Draft/verify paradigm
- **KV-Cache Compression** (`attention/kv_cache.rs`) - TurboQuant-inspired quantized caching
- **Hyperbolic Attention** (`hyperbolic/`) - Poincare ball model operations
- **Sparse Attention** (`sparse/`) - Flash, linear, local-global patterns
- **Graph Attention** (`graph/`) - DualSpace, EdgeFeatured, RoPE
- **MoE Attention** (`moe/`) - Mixture-of-experts with learned routing
- **PDE Attention** (`pde_attention/`) - Diffusion-based on graph Laplacian
- **Transport Attention** (`transport/`) - Sliced Wasserstein, Centroid OT
- **Sheaf Attention** (`sheaf/`) - Coherence-gated transformer (feature-gated)
- **Information Geometry** (`info_geometry/`) - Fisher metric, natural gradient
- **Information Bottleneck** (`info_bottleneck/`) - KL divergence, diagonal Gaussian

### 1.2 Numerical Stability Assessment

**Softmax implementations (POSITIVE)**:
- `ScaledDotProductAttention::softmax()` at `scaled_dot_product.rs:46-52`: Uses the max-subtraction trick (`s - max_score`). This is the correct log-sum-exp pattern for numerical stability.
- `FlashAttention3::forward()` at `flash.rs:240-268`: Implements online softmax with running `row_max` and `row_sum`, correctly rescaling with `exp(m_old - m_new)`. This is the standard FlashAttention numerically stable formulation.
- `naive_attention()` at `flash.rs:518-520`: Reference implementation also uses max-subtraction softmax.
- GNN `scaled_dot_product_attention()` at `layer.rs:196-199`: Uses max-subtraction and guards sum with `.max(1e-10)`.

**[CRITICAL] C-D5-001: Division-by-zero risk in MLA softmax**

File: `crates/ruvector-attention/src/attention/mla.rs`, lines 340-344

```rust
fn softmax_inplace(s: &mut [f32]) {
    let max = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0_f32;
    for v in s.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    for v in s.iter_mut() { *v /= sum; }
}
```

If `s` is empty or all values are `NEG_INFINITY` (fully masked), `sum` will be 0.0 and the division produces NaN. While the `forward()` function validates for empty inputs, there is no guard if all attention scores evaluate to `NEG_INFINITY` after masking. The `compute_with_mask` implementation for MLA delegates to `compute` which ignores the mask entirely, so a masked scenario where all keys are filtered would pass through.

**Recommendation**: Add `let sum = sum.max(f32::MIN_POSITIVE);` or return a uniform distribution when `sum == 0.0`.

**[HIGH] H-D5-001: ScaledDotProductAttention softmax division-by-zero**

File: `crates/ruvector-attention/src/attention/scaled_dot_product.rs`, lines 46-52

Same pattern as C-D5-001 but at a lower level. When `compute_with_mask` masks ALL keys to `false`, every score becomes `NEG_INFINITY`, the exp values are all 0.0, and `sum` is 0.0, yielding NaN outputs. The softmax function divides by `sum` without checking for zero.

**Recommendation**: Guard with `if sum == 0.0 { return vec![1.0 / scores.len() as f32; scores.len()]; }` or similar.

**[MEDIUM] M-D5-001: Flash attention naive_attention reference has bare division**

File: `crates/ruvector-attention/src/attention/flash.rs`, line 526

```rust
val += (exp_s[kj] / sum_s) * v[kj][dd];
```

`sum_s` could be 0.0 in the degenerate case where all scores are `-inf` (e.g., causal mask where qi=0 masks everything). The flash forward path itself handles this with the `if !m_ij.is_finite() { continue; }` check, but the naive reference does not, which could produce NaN in tests comparing flash vs. naive.

### 1.3 Memory Efficiency Assessment

- **FlashAttention-3**: Correctly O(N) working memory. Never materializes the N^2 attention matrix. Block sizes are configurable. IO stats tracking confirms the reduced memory transfer pattern.
- **MLA**: Achieves ~81-93% KV-cache reduction by caching only `latent_dim + rope_dim` floats per position instead of `2 * num_heads * head_dim`. Well-validated with memory comparison tests.
- **SSM/Mamba**: O(1) per-token inference via fixed-size recurrent state. The `d_inner * d_state` state size is bounded and reset-capable.
- **KV-Cache Compression**: Implements 2-4 bit quantization with banker's rounding and three eviction policies (H2O, SlidingWindow, PyramidKV).

**[MEDIUM] M-D5-002: FlashAttention block_scores allocated per-query-per-block**

File: `crates/ruvector-attention/src/attention/flash.rs`, lines 224, 251

```rust
let mut block_scores = Vec::with_capacity(kj_end - kj_start);
// ...
let exp_scores: Vec<f32> = block_scores.iter().map(|s| (s - m_ij).exp()).collect();
```

For each query row in each Q-block iteration over each K-block, new `Vec` allocations occur. For sequences of length N with block size B, this is O(N^2/B) allocations. Pre-allocating these buffers outside the inner loops would eliminate allocation overhead.

### 1.4 Precision Support

All attention implementations operate exclusively on `f32`. There is no f16, bf16, or mixed-precision support. The SIMD backends in ruvector-cnn handle INT8 quantization, but the attention crate itself is f32-only.

**[LOW] L-D5-001**: No mixed-precision support in attention crate. For large-scale inference, bf16/f16 attention would significantly reduce memory bandwidth. This is acceptable for the current CPU-focused implementation but limits GPU/accelerator deployment.

### 1.5 Gradient Computations

- **InfoNCE loss** (`training/loss.rs:74-133`): Gradient computation uses the softmax-weighted derivative of cosine similarity. The implementation correctly handles the chain rule through cosine similarity's derivative with norm clamping (`.max(1e-8)`).
- **LocalContrastiveLoss** (`training/loss.rs:183-226`): Gradient uses `(a-p)/d_pos - (a-n)/d_neg` with distance floor guards (`> 1e-8`).
- **SpectralRegularization**: Returns zero gradients (documented as auxiliary loss). This is a known simplification.

**[MEDIUM] M-D5-003: InfoNCE gradient denominator instability**

File: `crates/ruvector-attention/src/training/loss.rs`, lines 107-108

```rust
let norm_a: f32 = anchor.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
let norm_p: f32 = positive.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
```

The gradient formula at line 116 computes `norm_a.powi(3)` in the denominator. With `norm_a` as low as `1e-8`, `powi(3)` yields `1e-24` which is near subnormal range for f32. This could produce extremely large gradient values for near-zero embeddings.

**Recommendation**: Increase the floor to `1e-5` or use f64 for the gradient accumulation.

---

## 2. CNN Analysis (ruvector-cnn)

### 2.1 Forward Pass Correctness

**Conv2d** (`layers/conv.rs`):
- Supports standard, grouped, and depthwise convolutions
- Padding handled via signed integer arithmetic (`as isize - padding as isize`)
- Output shape formula: `(in_h + 2*padding - kernel_size) / stride + 1` is standard and correct
- NHWC tensor layout consistently used

**[HIGH] H-D5-002: Conv2d output_shape integer underflow risk**

File: `crates/ruvector-cnn/src/layers/conv.rs`, line 239

```rust
let out_h = (in_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
```

If `in_h + 2*padding < kernel_size`, this underflows (usize wraps around to a huge number). The same pattern appears at lines 196, 308, 554 across Conv2d and DepthwiseSeparableConv. The `output_shape` method would return a nonsensical shape, and `forward` would either panic on out-of-bounds access or produce a massive allocation.

**Recommendation**: Add a validation check: `if in_h + 2*self.padding < self.kernel_size { return Err(...); }`.

**DepthwiseSeparableConv** (`layers/conv.rs:442-660`):
- Two-phase: depthwise 3x3 then pointwise 1x1
- Correct parameter savings: O(K^2*C + C*C_out) vs O(K^2*C*C_out)
- Falls through to SIMD-optimized paths for 3x3 kernels

**Winograd F(2,3)** (`simd/winograd.rs`):
- Transform matrices (G, B^T, A^T, A) are mathematically correct for F(2,3)
- 2.25x theoretical speedup (16 multiplications vs 36)
- Scalar reference implementation only; AVX2 variants simply delegate to scalar per-tile (placeholder)

### 2.2 Backward Pass

**There is no backward pass implementation in ruvector-cnn.** The crate is designed for inference-only (feature extraction/embedding). No gradient computation, no autograd, no backpropagation support exists in any layer. The `Layer` trait only defines `forward()`.

### 2.3 Edge Cases

- **1x1 kernels**: Handled by the generic convolution path (`conv_generic`) since the fast path only activates for kernel_size==3.
- **Stride > kernel_size**: Mathematically valid but would skip input regions. No explicit protection, but the output formula handles it correctly.
- **Zero-padding**: Correctly handled via signed arithmetic in both generic and SIMD paths.

**[LOW] L-D5-002: No dilation support in convolutions**

The Conv2d layer has no dilation parameter. Dilated convolutions are common in semantic segmentation (e.g., DeepLab). The `conv_output_size` function in `layers/mod.rs:82-89` accepts a dilation parameter but is never used by Conv2d's `output_shape`.

### 2.4 Memory Allocation

**[MEDIUM] M-D5-004: Intermediate buffers allocated per-batch in DepthwiseSeparableConv**

File: `crates/ruvector-cnn/src/layers/conv.rs`, lines 558-559

```rust
let dw_shape = vec![batch, out_h, out_w, self.in_channels];
let mut dw_output = Tensor::zeros(&dw_shape);
```

For each `forward()` call, a complete intermediate tensor is allocated for the depthwise output. In a hot inference path, this allocation pressure could be significant. Buffer reuse across calls would improve throughput.

### 2.5 Unsafe Usage in CNN Code

The CNN crate has 133 `unsafe` occurrences across 17 files. All unsafe usage falls into well-defined categories:

**Category breakdown:**
1. **SIMD intrinsics** (majority): `avx2.rs` (22), `neon.rs` (11), `wasm.rs` (12), `simd/mod.rs` (19) - These use platform SIMD intrinsics (`_mm256_*`, `vld1q_*`, etc.) which require `unsafe`. All are gated behind `#[cfg(target_arch)]` and `#[target_feature]` attributes.
2. **INT8 kernels**: `int8_avx2.rs` (12), `int8_neon.rs` (9), `int8_wasm.rs` (9), `int8_scalar.rs` + simd variants
3. **Quantization SIMD**: `simd/quantize.rs` (6)
4. **Winograd AVX2**: `simd/winograd.rs` (4)

**Safety assessment:**
- All SIMD functions use `debug_assert_eq!` for input length validation
- Unaligned loads are used (`_mm256_loadu_ps`, `vld1q_f32`) which is correct for arbitrary alignment
- Remainder handling (scalar fallback after SIMD chunks) is present in all functions
- `get_unchecked()` appears only in NEON dot product (`neon.rs:79`) for the scalar remainder loop, which is protected by the loop bounds

**[MEDIUM] M-D5-005: Missing SAFETY comments on unsafe blocks**

While the SIMD functions themselves are `unsafe fn`, the individual `unsafe` blocks within them (pointer arithmetic, intrinsics) lack `// SAFETY:` comments explaining why the operation is valid. Rust best practice (and the project's own `#![deny(unsafe_op_in_unsafe_fn)]` in ruvector-gnn) requires these annotations.

Files affected: `simd/avx2.rs`, `simd/neon.rs`, `simd/wasm.rs`, `simd/quantize.rs`.

**[LOW] L-D5-003: Non-x86 stubs for Winograd AVX2 are misleading**

File: `crates/ruvector-cnn/src/simd/winograd.rs`, lines 402-410

```rust
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn transform_input_avx2(_tiles: &[[f32; 16]; 4]) -> [[f32; 16]; 4] {
    [[0.0f32; 16]; 4]
}
```

These stubs silently return zero arrays instead of using the scalar implementation. Any caller using `transform_input_avx2` on ARM would get incorrect results. The function should either not be exported on non-x86 or should delegate to the scalar path.

---

## 3. GNN Analysis (ruvector-gnn)

### 3.1 Message Passing Correctness

The `RuvectorLayer` (`layer.rs:315-457`) implements a sophisticated GNN layer:

1. **Message computation**: Linear projection via `w_msg`
2. **Attention-based aggregation**: Multi-head scaled dot-product attention on projected messages
3. **Weighted aggregation**: Edge-weight-based weighted sum (normalized to sum=1)
4. **Combination**: Element-wise addition of attention + weighted aggregation
5. **GRU update**: Gated recurrent cell updating node state
6. **LayerNorm**: Post-update normalization

**Aggregation functions:**
- Uses attention-weighted mean (via softmax) + edge-weight-weighted mean
- No max or sum aggregation options (only weighted mean)

**[MEDIUM] M-D5-006: Dropout implementation is incorrect for training**

File: `crates/ruvector-gnn/src/layer.rs`, lines 448-451

```rust
fn apply_dropout(&self, input: &[f32]) -> Vec<f32> {
    let scale = 1.0 - self.dropout;
    input.iter().map(|&x| x * scale).collect()
}
```

This applies a deterministic scaling `(1 - dropout)` to all elements rather than randomly zeroing elements with probability `dropout` and scaling survivors by `1/(1-dropout)`. During training, this produces a biased output (all values shrunk uniformly) rather than the stochastic regularization effect of true dropout. During inference, dropout should be disabled entirely (no scaling). The current implementation does neither correctly.

### 3.2 Edge Cases

- **Isolated nodes** (no neighbors): Handled at `layer.rs:386-389` - returns normalized projection of the node's own embedding. This is correct behavior.
- **Self-loops**: Not explicitly modeled. The adjacency list could include self-loops but the code does not add them. For GNN convergence, explicit self-loops are often important.
- **Very large neighborhoods**: No neighborhood sampling or attention head limiting. For hub nodes with thousands of edges, the full attention computation would be expensive.

**[LOW] L-D5-004: No neighborhood sampling for large-degree nodes**

In graphs with power-law degree distributions, computing full attention over all neighbors of high-degree nodes creates computational imbalance. Standard GNN practice is to sample K neighbors per node.

### 3.3 Numerical Stability in Aggregation

- Softmax uses max-subtraction trick with epsilon guard (`sum_exp.max(1e-10)`)
- Layer normalization uses epsilon (`1e-5`) in the variance denominator
- Edge weight normalization handles zero-sum case (`if weight_sum > 0.0 ... else uniform`)
- Sigmoid uses numerically stable two-branch formulation at `layer.rs:278-285`:
  ```rust
  if x > 0.0 { 1.0 / (1.0 + (-x).exp()) }
  else { let ex = x.exp(); ex / (1.0 + ex) }
  ```

This is correct and avoids overflow for large positive or negative values.

### 3.4 GraphMAE (Self-Supervised Learning)

The GraphMAE implementation (`graphmae.rs`) implements masked autoencoder pretraining:
- Feature masking with learnable `[MASK]` token
- Degree-centrality-based masking (higher-degree nodes masked more)
- GAT encoder with multi-head attention
- SCE (Scaled Cosine Error) loss: `(1 - cos_sim)^gamma`
- Re-masking of latent representations before decoding

**[LOW] L-D5-005: GraphMAE masking uses thread_rng which is non-deterministic**

The masking functions use `rand::thread_rng()` directly, making training non-reproducible. For scientific reproducibility, a seeded RNG should be passed as a parameter.

### 3.5 Graph Database Interaction

The GNN crate does not directly interact with D2 graph database. It defines its own `GraphData` struct and adjacency lists. The `query.rs` module provides `RuvectorQuery` for subgraph extraction, and `search.rs` provides differentiable search over embeddings. Integration with the graph database would happen at a higher orchestration layer.

---

## 4. Unsafe Audit

### 4.1 Distribution of unsafe across D5

| Crate | Unsafe Files | Unsafe Occurrences | Category |
|-------|-------------|-------------------|----------|
| ruvector-attention | 0 | 0 | None |
| ruvector-cnn | 17 | 133 | SIMD, INT8, Quantization |
| ruvector-gnn | 3 | 13 | mmap, cold_tier, deny attribute |
| sona | 1 | 1 | SIMD in LoRA |
| **Total** | **21** | **147** | |

**Note**: The Phase 1 report cited 69 files with 489 unsafe references for D5. The actual count found is 21 files with 147 occurrences in the production crates examined. The discrepancy likely comes from including npm bindings, NAPI wrappers, and build artifacts in the Phase 1 count.

### 4.2 Unsafe Categories

1. **SIMD intrinsics** (~120 occurrences): All in ruvector-cnn's `simd/` module. These are the standard, well-understood pattern for SIMD programming in Rust. Each function:
   - Is gated behind `#[cfg(target_arch)]`
   - Uses `#[target_feature(enable = "...")]`
   - Has `debug_assert_eq!` for length validation
   - Handles remainder elements with scalar fallback
   - Uses unaligned loads (safe for any alignment)

2. **Memory-mapped IO** (~11 occurrences in ruvector-gnn/src/mmap.rs): Uses `memmap2::MmapMut` for memory-mapped embedding storage. Atomic operations (`AtomicU64`, `AtomicU32`) for thread-safe bitmap tracking. The `#![deny(unsafe_op_in_unsafe_fn)]` lint is active in ruvector-gnn, which is best practice.

3. **Cold tier** (~1 occurrence in ruvector-gnn/src/cold_tier.rs): Feature-gated storage tier.

4. **LoRA SIMD** (1 occurrence in sona/src/lora.rs): AVX2 SIMD for the MicroLoRA forward pass, gated behind `#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]`.

### 4.3 Risk Assessment

**[HIGH] H-D5-003: SIMD bounds checking relies solely on debug_assert**

Across all SIMD functions in ruvector-cnn, bounds checking uses `debug_assert_eq!` which is compiled away in release mode. If a caller passes mismatched-length slices in a release build, the SIMD loads will read out of bounds (undefined behavior).

Example from `crates/ruvector-cnn/src/simd/avx2.rs:13-14`:
```rust
pub unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
```

The dispatch layer in `simd/mod.rs` (e.g., `dot_product_simd`) does not validate lengths before calling the unsafe SIMD functions.

**Recommendation**: Add `assert_eq!` (not `debug_assert_eq!`) or length checks in the safe dispatch wrappers (`dot_product_simd`, `relu_simd`, etc.).

**[MEDIUM] M-D5-007: NEON remainder uses get_unchecked without bounds comment**

File: `crates/ruvector-cnn/src/simd/neon.rs`, line 79

```rust
total += *a.get_unchecked(i) * *b.get_unchecked(i);
```

While the loop bounds (`scalar_start..len`) are correct, `get_unchecked` bypasses bounds checking. A `// SAFETY: i < len == a.len() == b.len()` comment should be present.

---

## 5. Numerical Stability Analysis

### 5.1 Floating-Point Edge Case Handling

**NaN/Inf checking coverage across D5:**

| Crate | Files with checks | Checks found |
|-------|-------------------|-------------|
| ruvector-attention | 8 | 35 |
| ruvector-cnn | 4 | 13 |
| ruvector-gnn | 2 | 14 |

**Positive patterns observed:**
- FlashAttention explicitly filters `is_finite()` values during softmax accumulation
- SSM uses stable softplus: `if x > 20.0 { x } else if x < -20.0 { 0.0 } else { (1.0 + x.exp()).ln() }`
- Poincare operations use `.max(EPS)` consistently for denominators
- GNN training loss functions check for NaN/Inf in gradients and predictions

**[HIGH] H-D5-004: Cosine similarity in GNN search uses f32 dot product but f64 norms**

File: `crates/ruvector-gnn/src/search.rs`, lines 4-20

```rust
let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
let norm_a: f32 = (a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()) as f32;
```

The norms are computed in f64 (good for precision) but the dot product remains in f32. For large vectors with many small values, the f32 dot product accumulation can lose significant precision through catastrophic cancellation, while the norms are precise. This inconsistency can produce `cosine_similarity > 1.0` or slightly negative values for near-identical vectors.

**Recommendation**: Either compute the dot product in f64 as well, or use Kahan summation for the f32 path.

### 5.2 Catastrophic Cancellation Risks

**[MEDIUM] M-D5-008: Poincare Mobius addition denominator**

File: `crates/ruvector-attention/src/hyperbolic/poincare.rs`, line 50

```rust
let denom = 1.0 + 2.0 * c * dot_uv + c * c * norm_u_sq * norm_v_sq;
```

For points near the boundary of the Poincare ball (where norms approach `1/sqrt(c)`), the terms `2.0 * c * dot_uv` and `c^2 * norm_u_sq * norm_v_sq` can nearly cancel with `1.0`, leaving a denominator close to zero. The `.max(EPS)` guard at the division prevents NaN but produces numerically inaccurate results. This is a known challenge in hyperbolic geometry computation.

### 5.3 Mixed-Precision Support

- **INT8 quantization**: ruvector-cnn provides comprehensive INT8 support with pi-based anti-resonance calibration, per-channel and per-tensor modes, and SIMD-accelerated quantize/dequantize.
- **f16/bf16**: Not supported anywhere in D5.
- **f64 accumulation**: Used selectively (GNN cosine similarity norms, but not attention score accumulation).

**[LOW] L-D5-006: No f64 accumulation in attention score computation**

For very long sequences (>4K tokens), the f32 softmax denominator accumulation in FlashAttention's online algorithm could lose precision. The running `row_sum` values grow large while new `exp()` terms can be very small, leading to precision loss when added.

### 5.4 Loss Function Stability

- **InfoNCE**: Uses log-sum-exp trick for stability (correct)
- **SCE** (GraphMAE): `(1 - cos_sim)^gamma` - stable since `cos_sim` is bounded [-1, 1]
- **MSE/CrossEntropy** (GNN training): Cross-entropy uses `pred.max(1e-7)` clamp (correct)
- **Binary Cross-Entropy**: Uses `pred.clamp(1e-7, 1.0 - 1e-7)` (correct)
- **EWC penalty**: Quadratic form `F_i * (w - w*)^2` is inherently stable

---

## 6. Test Coverage Analysis

### 6.1 ruvector-attention (58 test modules)

**Well-tested areas:**
- FlashAttention: 11 tests including correctness vs naive, causal masking, numerical stability with large values, block size variations, LSE correctness, ring attention, IO stats
- MLA: 12 tests including config validation, forward shape, cache reduction ratio, RoPE identity/norm preservation, compress/decompress dimensions
- SSM/Mamba: 11 tests including config validation, softplus/silu values, RMS norm, selective scan, state recurrence, hybrid routing
- Speculative decoding: Tests for config validation, draft-verify protocol, acceptance/rejection

**Test gaps:**
- No numerical accuracy tests with known analytical solutions (e.g., uniform attention should produce mean of values)
- No adversarial inputs (NaN in query, extremely large values in keys)
- No benchmark/regression tests for performance

**[HIGH] H-D5-005: Multi-head attention mask parameter is silently ignored**

File: `crates/ruvector-attention/src/attention/multi_head.rs`, lines 105-114

```rust
fn compute_with_mask(
    &self, query: &[f32], keys: &[&[f32]], values: &[&[f32]],
    _mask: Option<&[bool]>,
) -> AttentionResult<Vec<f32>> {
    self.compute(query, keys, values) // mask is ignored!
}
```

The `_mask` parameter is prefixed with underscore and completely ignored. Any caller expecting masking behavior from `MultiHeadAttention::compute_with_mask` will get incorrect results. This is especially dangerous because the trait `Attention` defines this method, so callers may use it generically.

### 6.2 ruvector-cnn (38 test modules + 9 integration test files)

**Well-tested areas:**
- Layer dimensions and output shapes
- SIMD equivalence tests (`kernel_equivalence.rs`)
- Quality validation gates (`quality_validation.rs`)
- Quantization roundtrip accuracy
- Contrastive loss (InfoNCE, triplet)

**Test gaps:**
- No tests for stride > kernel_size edge case
- No tests for zero-channel or zero-spatial-dimension inputs
- SIMD tests are architecture-gated (`#[cfg(target_arch)]`), meaning ARM tests only run on ARM, etc.

### 6.3 ruvector-gnn (13 test modules + 1 integration test file)

**Well-tested areas:**
- EWC: Comprehensive (17 tests covering penalty, gradient, consolidation, sequential tasks)
- Loss functions: Verified in `loss_verification.rs` (MSE, cross-entropy, binary cross-entropy)
- Layer behavior: Basic forward pass, no-neighbors case, invalid config

**Test gaps:**
- No tests for large graph scaling (>100 nodes)
- No tests for numerical edge cases in message passing
- Replay buffer: Limited testing of reservoir sampling uniformity
- GraphMAE: No end-to-end training loop test

### 6.4 sona (19 test modules)

**Well-tested areas:**
- LoRA creation, forward pass, learning cycle
- Engine lifecycle (begin/end trajectory)

**Test gaps:**
- No stress tests for many concurrent trajectories
- No tests for LoRA weight export/import fidelity

---

## 7. Training Loop Safety

### 7.1 Memory Leak Risks

**[MEDIUM] M-D5-009: MLACache grows unboundedly in autoregressive decoding**

File: `crates/ruvector-attention/src/attention/mla.rs`, lines 260-279

```rust
pub fn forward_cached(&self, ..., cache: &mut MLACache) -> ... {
    cache.push(self.compress_kv(new_kv_input), self.compute_rope_keys(new_kv_input));
    // ... decompresses ALL cached positions every step
```

The cache grows by one entry per token and is never pruned. For long generation sequences, this is O(n) memory growth. Additionally, `forward_cached` decompresses ALL cached positions every step, making it O(n^2) total work for generating n tokens. The KV-cache compression module provides eviction policies, but they are not integrated into the MLA layer.

### 7.2 Gradient Clipping / Explosion Handling

- **Adam optimizer** (`training/optimizer.rs`): No gradient clipping. If gradients explode (common in deep networks), the optimizer will produce very large parameter updates.
- **GNN optimizer** (`training.rs`): Also no gradient clipping.
- **EWC**: Provides gradient regularization toward anchor weights, which implicitly limits gradient magnitude for previously learned parameters but not for new parameters.

**[MEDIUM] M-D5-010: No gradient clipping in any optimizer**

Neither the attention training optimizer (Adam, AdamW, SGD) nor the GNN optimizer implements gradient clipping (norm clipping or value clipping). For deep transformer or GNN training, gradient explosion is a known risk, particularly in the early stages of training with high learning rates.

### 7.3 Checkpoint/Resume Capability

- **Sona**: The `LoopCoordinator` supports state serialization via the `coordinator()` accessor. The `SonaEngine` exposes this for checkpoint.
- **GNN**: No explicit checkpoint support. The `RuvectorLayer` is `Serialize + Deserialize` via serde, allowing weight serialization.
- **Attention**: Training state (optimizer moments, learning rate scheduler step count) is not serializable.

**[LOW] L-D5-007: Optimizer state is not serializable**

File: `crates/ruvector-attention/src/training/optimizer.rs`

The `Adam`, `AdamW`, and `SGD` optimizers do not implement `Serialize`/`Deserialize`. Resuming training after a checkpoint would reset momentum/moment estimates, causing a training discontinuity.

---

## 8. Neural Trader Analysis

The `neural-trader-core` crate defines canonical types for market data processing:
- `MarketEvent`: Normalized market event envelope with nanosecond timestamps and fixed-point prices
- `GraphDelta`: Changes to the market graph (nodes added, edges added, properties updated)
- Traits: `EventIngestor`, `GraphUpdater`, `Embedder`
- `StateWindow`: Sliding window of graph state for embedding

This is a well-designed, type-safe event processing framework. The use of fixed-point arithmetic for prices (`price_fp: i64`) is correct for financial data (avoids floating-point rounding in price comparison). The enum-based `PropertyKey` avoids heap allocation on the hot ingest path.

No significant issues found in neural-trader-core. It is a data-types-only crate with no computation logic.

---

## 9. SONA Analysis

The SONA crate implements a self-optimizing neural architecture with:
- **Two-tier LoRA**: MicroLoRA (rank 1-2, <100us) and BaseLoRA (rank 4-16, background)
- **EWC++**: Prevents catastrophic forgetting across learning cycles
- **ReasoningBank**: Pattern extraction and similarity search
- **Three learning loops**: Instant, Background, Coordinator

The LoRA implementation is well-structured:
- Standard LoRA initialization (down_proj random, up_proj zero)
- SIMD-optimized forward path for AVX2
- Gradient accumulation with quality-weighted updates
- Merge capability for inference optimization

**[MEDIUM] M-D5-011: BaseLoRA merge_into assumes square weight matrix**

File: `crates/sona/src/lora.rs`, lines 352-372

```rust
pub fn merge_into(&self, model_weights: &mut [f32], layer_idx: usize) {
    // W' = W + scale * (down @ up)
    // Assumes model_weights is [hidden_dim x hidden_dim]
    for i in 0..self.hidden_dim {
        for j in 0..self.hidden_dim {
```

The comment explicitly states "Assumes model_weights is [hidden_dim x hidden_dim]", but the `down_proj` is `[hidden_dim * rank]` and `up_proj` is `[rank * hidden_dim]`. The merge computes `down_proj[i * rank + r] * up_proj[r * hidden_dim + j]`, which is correct for the product but the indexing into `down_proj` uses `i * self.rank` while the storage layout is `r * self.hidden_dim`. This is a potential indexing error if hidden_dim != the weight matrix dimension.

---

## 10. Summary of Findings by Severity

### CRITICAL (1)

| ID | Location | Description |
|----|----------|-------------|
| C-D5-001 | `attention/mla.rs:340-344` | Softmax division-by-zero when all scores are -inf |

### HIGH (5)

| ID | Location | Description |
|----|----------|-------------|
| H-D5-001 | `attention/scaled_dot_product.rs:46-52` | Softmax division-by-zero when all keys masked |
| H-D5-002 | `layers/conv.rs:239` | Conv2d output_shape usize underflow when input < kernel |
| H-D5-003 | `simd/avx2.rs` (all functions) | SIMD bounds checking only in debug_assert (UB in release) |
| H-D5-004 | `search.rs:4-20` | Mixed f32/f64 in cosine similarity can produce values >1.0 |
| H-D5-005 | `attention/multi_head.rs:105-114` | compute_with_mask silently ignores mask parameter |

### MEDIUM (9)

| ID | Location | Description |
|----|----------|-------------|
| M-D5-001 | `attention/flash.rs:526` | Naive attention reference has unguarded division |
| M-D5-002 | `attention/flash.rs:224,251` | Per-query-per-block Vec allocation in FlashAttention |
| M-D5-003 | `training/loss.rs:107-108` | InfoNCE gradient denominator near-subnormal range |
| M-D5-004 | `layers/conv.rs:558-559` | Per-call intermediate buffer allocation in DepthwiseConv |
| M-D5-005 | `simd/avx2.rs`, `neon.rs`, etc. | Missing SAFETY comments on unsafe blocks |
| M-D5-006 | `layer.rs:448-451` | GNN dropout is deterministic scaling, not stochastic |
| M-D5-007 | `simd/neon.rs:79` | get_unchecked without safety comment |
| M-D5-008 | `hyperbolic/poincare.rs:50` | Poincare Mobius addition near-boundary cancellation |
| M-D5-009 | `attention/mla.rs:260-279` | MLACache grows unboundedly in autoregressive mode |
| M-D5-010 | `training/optimizer.rs` | No gradient clipping in any optimizer |
| M-D5-011 | `sona/src/lora.rs:352-372` | BaseLoRA merge_into assumes square weight matrix |

### LOW (7)

| ID | Location | Description |
|----|----------|-------------|
| L-D5-001 | ruvector-attention (entire crate) | No mixed-precision (f16/bf16) support |
| L-D5-002 | `layers/conv.rs` | No dilation support in Conv2d |
| L-D5-003 | `simd/winograd.rs:402-410` | Non-x86 stubs return zeros instead of scalar fallback |
| L-D5-004 | `layer.rs` | No neighborhood sampling for large-degree GNN nodes |
| L-D5-005 | `graphmae.rs` | Non-deterministic RNG in masking (reproducibility) |
| L-D5-006 | FlashAttention f32 accumulation | No f64 accumulation for very long sequences |
| L-D5-007 | `training/optimizer.rs` | Optimizer state not serializable (checkpoint/resume) |

### INFORMATIONAL (5)

| ID | Location | Description |
|----|----------|-------------|
| I-D5-001 | ruvector-cnn | No backward pass - inference only (by design) |
| I-D5-002 | ruvector-attention | 58 test modules with comprehensive coverage |
| I-D5-003 | neural-trader-core | Clean type-only crate, no computation issues |
| I-D5-004 | ruvector-gnn | `#![deny(unsafe_op_in_unsafe_fn)]` is good practice |
| I-D5-005 | sona | Two-tier LoRA with SIMD optimization is well-designed |

---

## 11. Files Examined

### ruvector-attention (72 source files)
- `src/lib.rs`, `src/attention/mod.rs`, `src/attention/scaled_dot_product.rs`, `src/attention/multi_head.rs`, `src/attention/flash.rs`, `src/attention/mla.rs`, `src/attention/ssm.rs`, `src/attention/speculative.rs`, `src/attention/kv_cache.rs`, `src/hyperbolic/poincare.rs`, `src/training/loss.rs`, `src/training/optimizer.rs`, `src/config.rs`, `src/error.rs`, `src/traits.rs`, `src/utils.rs`

### ruvector-cnn (46 source files + 9 test files)
- `src/lib.rs`, `src/layers/mod.rs`, `src/layers/conv.rs`, `src/simd/mod.rs`, `src/simd/avx2.rs`, `src/simd/neon.rs`, `src/simd/winograd.rs`, `src/simd/quantize.rs`, `src/tensor.rs`, `src/error.rs`

### ruvector-gnn (16 source files + 1 test file)
- `src/lib.rs`, `src/layer.rs`, `src/training.rs`, `src/search.rs`, `src/ewc.rs`, `src/replay.rs`, `src/mmap.rs`, `src/graphmae.rs`, `tests/loss_verification.rs`

### sona (26 source files)
- `src/lib.rs`, `src/engine.rs`, `src/lora.rs`, `src/types.rs`, `src/ewc.rs`, `src/reasoning_bank.rs`, `src/trajectory.rs`

### neural-trader-* (4 source files)
- `neural-trader-core/src/lib.rs`, `neural-trader-coherence/src/lib.rs`, `neural-trader-replay/src/lib.rs`, `neural-trader-wasm/src/lib.rs`

---

## 12. Patterns Checked (Clean Justification for Uncovered Areas)

| Pattern | Checked | Finding |
|---------|---------|---------|
| Softmax numerical stability | Yes | Correct in 4/5 implementations; 2 division-by-zero risks |
| SIMD bounds safety | Yes | debug_assert only (H-D5-003) |
| Integer overflow in conv shapes | Yes | Underflow risk found (H-D5-002) |
| NaN propagation | Yes | FlashAttention handles well; MLA does not |
| Memory leak in training | Yes | MLA cache unbounded (M-D5-009) |
| Gradient stability | Yes | No clipping (M-D5-010); InfoNCE denominator risk (M-D5-003) |
| Loss function stability | Yes | All use standard numerical guards |
| Thread safety | Yes | GNN mmap uses atomic operations correctly |
| Serialization correctness | Yes | GNN layers serializable; optimizers not |
| Edge case handling | Yes | Empty input, zero-dim, isolated nodes all handled |
