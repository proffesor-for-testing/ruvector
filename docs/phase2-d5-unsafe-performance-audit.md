# Phase 2: D5 Neural/ML Domain -- Unsafe & Performance Audit

**Auditor**: V3 QE Security Reviewer (claude-opus-4-6)
**Date**: 2026-03-29
**Domain**: D5 Neural/ML (ruvector-attention, ruvector-cnn, ruvector-gnn, neural-trader-*, sona)
**Scope**: 16 crates, 133+ unsafe references across production code

---

## Executive Summary

D5 is the highest-unsafe-count production domain in the RuVector monorepo. The unsafe code is concentrated in **ruvector-cnn** (133 occurrences in 11 source files) with secondary clusters in **ruvector-gnn** (13 occurrences in 3 files) and **sona** (1 occurrence). Notably, **ruvector-attention** -- the largest crate by source file count (69 .rs files) -- contains **zero** unsafe code, which is commendable.

**Critical findings**: 5 HIGH-risk issues, 8 MEDIUM-risk issues, 4 LOW-risk issues.

The same `debug_assert_eq!` pattern found in D1 is pervasive here: 40+ boundary checks across SIMD hot paths use `debug_assert` instead of `assert`, meaning they are stripped in release builds. While the SIMD code is architecturally gated behind `#[target_feature]` and `#[cfg]`, the missing runtime assertions on slice lengths create out-of-bounds read/write risk when callers pass mismatched sizes.

---

## Part 1: Unsafe Audit

### 1.1 Crate-Level Unsafe Census

| Crate | Files w/ unsafe | Unsafe Refs | Category |
|-------|----------------|-------------|----------|
| ruvector-cnn (src/simd/avx2.rs) | 1 | 22 | SIMD (AVX2/AVX-512) |
| ruvector-cnn (src/simd/mod.rs) | 1 | 19 | SIMD dispatch |
| ruvector-cnn (src/simd/wasm.rs) | 1 | 12 | SIMD (WASM) |
| ruvector-cnn (src/simd/neon.rs) | 1 | 11 | SIMD (NEON) |
| ruvector-cnn (src/kernels/int8_avx2.rs) | 1 | 12 | INT8 SIMD (AVX2) |
| ruvector-cnn (src/kernels/int8_neon.rs) | 1 | 9 | INT8 SIMD (NEON) |
| ruvector-cnn (src/kernels/int8_wasm.rs) | 1 | 9 | INT8 SIMD (WASM) |
| ruvector-cnn (src/simd/quantize.rs) | 1 | 6 | Quantization + transmute |
| ruvector-cnn (src/int8/kernels/simd.rs) | 1 | 5 | INT8 SIMD stub |
| ruvector-cnn (src/simd/winograd.rs) | 1 | 4 | Winograd AVX2 |
| ruvector-cnn (src/layers/quantized_conv2d.rs) | 1 | 2 | Layer dispatch |
| ruvector-gnn (src/mmap.rs) | 1 | 11 | Raw pointer, mmap, Send/Sync |
| ruvector-gnn (src/cold_tier.rs) | 1 | 1 | Raw pointer (f32->u8 cast) |
| ruvector-gnn (src/lib.rs) | 1 | 1 | `deny(unsafe_op_in_unsafe_fn)` |
| sona (src/lora.rs) | 1 | 1 | SIMD (AVX2) |
| ruvector-attention | 0 | 0 | None (safe-only) |
| neural-trader-* | 0 | 0 | None (safe-only) |

**Total**: 147 unsafe references in 15 files across 4 crates.

---

### 1.2 Top 15 Files: Detailed Unsafe Block Analysis

#### FILE 1: `crates/ruvector-cnn/src/simd/avx2.rs` (22 unsafe refs, 814 lines)

| Line(s) | What it does | Category | SAFETY comment? | Risk |
|---------|-------------|----------|-----------------|------|
| 12-37 | `dot_product_avx2_fma`: AVX2+FMA dot product with `_mm256_loadu_ps`, `_mm256_fmadd_ps`, horizontal sum | SIMD | No | MEDIUM |
| 43-69 | `dot_product_avx2`: AVX2 dot product without FMA, same pattern | SIMD | No | MEDIUM |
| 75-94 | `dot_product_avx512`: AVX-512 dot product | SIMD | No | LOW |
| 100-116 | `relu_avx2`: AVX2 max(0, x) with `_mm256_loadu_ps`/`_mm256_storeu_ps` | SIMD | No | LOW |
| 121-137 | `relu6_avx2`: AVX2 clamp(0, 6) | SIMD | No | LOW |
| 143-184 | `batch_norm_avx2`: Batch norm with pre-computed scale/shift; heap allocs `vec![0.0; channels]` inside unsafe fn | SIMD | No | MEDIUM |
| 201-350 | `conv_3x3_avx2_fma`: **Critical path**. 4x unrolled FMA conv with 24 `get_unchecked` calls on `input` and `kernel` slices | SIMD + raw indexing | No | **HIGH** |
| 360-499 | `conv_3x3_avx2`: Same as above without FMA, 24 `get_unchecked` calls | SIMD + raw indexing | No | **HIGH** |
| 508-623 | `depthwise_conv_3x3_avx2`: Depthwise conv with kernel cache pre-load, 1 `get_unchecked` | SIMD + raw indexing | No | MEDIUM |
| 631-667 | `global_avg_pool_avx2`: Global average pooling | SIMD | No | LOW |
| 675-725 | `max_pool_2x2_avx2`: Max pooling 2x2 | SIMD | No | LOW |

**Findings for avx2.rs**:

- **F-D5-001 (HIGH)**: `conv_3x3_avx2_fma` and `conv_3x3_avx2` (lines 254-310, 408-462) use `get_unchecked` extensively on both `input` and `kernel` slices with computed indices. The index expression `(oc_idx * in_c + ic_base) * 9 + kernel_offset` depends on caller-supplied `in_c`, `out_c`, and kernel dimensions. There is **no bounds validation at function entry** -- only a `debug_assert_eq!` in the dispatch layer (`mod.rs` lines 35, 71, etc.) that is stripped in release builds. If the caller passes `out_c` that exceeds the kernel slice length, this is an out-of-bounds read.

- **F-D5-002 (MEDIUM)**: All 7 functions in this file use `debug_assert_eq!` for input/output length validation (lines 13, 44, 76, 101, 122, 153). These are elided in `--release`. The dispatch functions in `mod.rs` (which the caller actually invokes) also only use `debug_assert_eq!`. This means a release-build caller passing mismatched slice lengths will get silent UB.

- **F-D5-003 (LOW)**: `batch_norm_avx2` (line 156-157) allocates `vec![0.0; channels]` twice inside an `unsafe fn`. This is not a safety issue per se, but heap allocation inside a hot path hurts performance and could trigger UB if the allocator returns null (though Rust's allocator panics on OOM).

- **No SAFETY comments** are present anywhere in this file.

---

#### FILE 2: `crates/ruvector-cnn/src/simd/mod.rs` (19 unsafe refs, 387 lines)

| Line(s) | What it does | Category | SAFETY comment? | Risk |
|---------|-------------|----------|-----------------|------|
| 39-66 | `dot_product_simd`: Dispatch to AVX512/AVX2+FMA/AVX2/NEON/WASM/scalar based on CPU features | Dispatch | No | MEDIUM |
| 71-98 | `relu_simd`: Same dispatch pattern for ReLU | Dispatch | No | MEDIUM |
| 103-131 | `relu6_simd`: Same dispatch pattern for ReLU6 | Dispatch | No | MEDIUM |
| 135-174 | `batch_norm_simd`: Same dispatch for batch norm | Dispatch | No | MEDIUM |
| 178-236 | `conv_3x3_simd`: Same dispatch for 3x3 convolution | Dispatch | No | MEDIUM |
| 240-279 | `depthwise_conv_3x3_simd`: Same dispatch for depthwise conv | Dispatch | No | MEDIUM |
| 283-311 | `global_avg_pool_simd`: Same dispatch for global avg pool | Dispatch | No | LOW |
| 315-350 | `max_pool_2x2_simd`: Same dispatch for max pool | Dispatch | No | LOW |

**Findings for mod.rs**:

- **F-D5-004 (MEDIUM)**: The dispatch functions are the public API. They call `unsafe { platform_specific_fn(...) }` directly. The only precondition checks are `debug_assert_eq!` which are stripped in release. Example at line 39: `dot_product_simd` calls `avx2::dot_product_avx2_fma(a, b)` but `a.len() == b.len()` is only a `debug_assert` in the called function. A release-build caller can trigger UB by passing slices of different lengths.

---

#### FILE 3: `crates/ruvector-cnn/src/simd/neon.rs` (11 unsafe refs, 534 lines)

| Line(s) | What it does | Category | SAFETY comment? | Risk |
|---------|-------------|----------|-----------------|------|
| 18-82 | `dot_product_neon`: 4x unrolled NEON dot product with `vfmaq_f32`, `vld1q_f32` | SIMD (NEON) | No | **HIGH** |
| 92-109 | `relu_neon`: NEON ReLU | SIMD | No | LOW |
| 115-134 | `relu6_neon`: NEON ReLU6 | SIMD | No | LOW |
| 144-199 | `batch_norm_neon`: NEON batch norm, heap allocs inside unsafe fn | SIMD | No | MEDIUM |
| 209-289 | `conv_3x3_neon`: NEON 3x3 convolution, processes 4 output channels | SIMD | No | MEDIUM |
| 299-372 | `depthwise_conv_3x3_neon`: NEON depthwise convolution | SIMD | No | MEDIUM |
| 382-431 | `global_avg_pool_neon`: NEON global average pooling | SIMD | No | LOW |
| 437-492 | `max_pool_2x2_neon`: NEON max pooling 2x2 | SIMD | No | LOW |

**Findings for neon.rs**:

- **F-D5-005 (HIGH)**: `dot_product_neon` (line 79) uses `get_unchecked` in the scalar remainder loop: `*a.get_unchecked(i) * *b.get_unchecked(i)`. This is the **exact same D1 pattern** (NEON `debug_assert_eq!` instead of `assert_eq!`). The precondition `a.len() == b.len()` at line 19 is only `debug_assert_eq!`. In release builds, if `a.len() != b.len()`, the remainder loop at line 78 iterates with `scalar_start..len` where `len = a.len()`, but accesses `b.get_unchecked(i)` which may be out of bounds if `b` is shorter. This is a **buffer overread**.

- **F-D5-006 (MEDIUM)**: `batch_norm_neon` (lines 158-159) allocates `vec![0.0; channels]` twice inside the unsafe function. Same pattern as AVX2.

- **No SAFETY comments** in the entire file.

---

#### FILE 4: `crates/ruvector-cnn/src/simd/wasm.rs` (12 unsafe refs, 551 lines)

| Line(s) | What it does | Category | SAFETY comment? | Risk |
|---------|-------------|----------|-----------------|------|
| 25-30 | `dot_product_wasm`: WASM SIMD dot product with `v128_load` | SIMD (WASM) | No | MEDIUM |
| 63-67 | `relu_wasm`: WASM ReLU with `v128_load`/`v128_store` | SIMD | No | LOW |
| 89-93 | `relu6_wasm`: WASM ReLU6 | SIMD | No | LOW |
| 145-151 | `batch_norm_wasm`: WASM batch norm, heap alloc inside hot path | SIMD | No | MEDIUM |
| 224-236 | `conv_3x3_wasm`: WASM 3x3 conv with `v128_load` on local array | SIMD | No | MEDIUM |
| 313-327 | `depthwise_conv_3x3_wasm`: WASM depthwise conv | SIMD | No | LOW |
| 370-410 | `global_avg_pool_wasm`: WASM global avg pool | SIMD | No | LOW |
| 443-455 | `max_pool_2x2_wasm`: WASM max pool 2x2 | SIMD | No | LOW |

**Findings for wasm.rs**:

- **F-D5-007 (MEDIUM)**: Same `debug_assert_eq!` pattern as all other SIMD files. All length checks at lines 17, 55, 80, 120, 121 are debug-only. The unsafe blocks assume the caller has validated inputs.

- The WASM SIMD code uses `v128_load` on `a[idx..].as_ptr() as *const v128`, which relies on Rust's bounds check on the slice sub-expression `a[idx..]` being present. Since this is bounds-checked (not `get_unchecked`), the WASM path is actually **safer** than the AVX2 and NEON paths for the load operations themselves.

---

#### FILE 5: `crates/ruvector-cnn/src/simd/quantize.rs` (6 unsafe refs, 559 lines)

| Line(s) | What it does | Category | SAFETY comment? | Risk |
|---------|-------------|----------|-----------------|------|
| 330-374 | `quantize_batch_avx2`: AVX2 batch quantization. Uses `transmute` at line 361 | SIMD + transmute | No | **HIGH** |
| 379-409 | `dequantize_batch_avx2`: AVX2 batch dequantization. Uses `transmute` at line 394 | SIMD + transmute | No | **HIGH** |
| 423-428 | `quantize_simd`: Dispatch wrapper | Dispatch | No | LOW |
| 439-441 | `dequantize_simd`: Dispatch wrapper | Dispatch | No | LOW |

**Findings for quantize.rs**:

- **F-D5-008 (HIGH)**: Two `std::mem::transmute` calls:
  - Line 361: `let i32_array: [i32; 8] = std::mem::transmute(i32_vals);` -- transmutes `__m256i` to `[i32; 8]`. This is the standard pattern for extracting AVX2 lanes but relies on `__m256i` being exactly 32 bytes. This is correct for the `x86_64` ABI where `__m256i` is indeed a 256-bit type. **Risk**: LOW in practice but should use `_mm256_storeu_si256` followed by a read from the buffer instead, which avoids `transmute` entirely.
  - Line 394: `let i32_vals: __m256i = std::mem::transmute(i32_array);` -- the reverse direction. Same analysis applies. Should use `_mm256_loadu_si256(i32_array.as_ptr() as *const __m256i)` instead.

- These transmutes are **between same-sized types** (both 32 bytes), so they are technically sound on x86_64. However, `transmute` is the most dangerous `unsafe` operation and should be avoided when safer alternatives exist. The `_mm256_storeu_si256`/`_mm256_loadu_si256` alternatives perform the same operation with less footgun surface.

---

#### FILE 6: `crates/ruvector-cnn/src/kernels/int8_avx2.rs` (12 unsafe refs, 587 lines)

| Line(s) | What it does | Category | SAFETY comment? | Risk |
|---------|-------------|----------|-----------------|------|
| 23-66 | `dot_product_int8_avx2`: AVX2 INT8 dot product using `_mm256_maddubs_epi16` cascade | SIMD | Yes (lines 18-20) | MEDIUM |
| 84-230 | `conv2d_int8_avx2`: AVX2 INT8 2D convolution, processes 8 output channels | SIMD | Yes (lines 80-81) | MEDIUM |
| 240-339 | `depthwise_conv2d_int8_avx2`: AVX2 INT8 depthwise convolution | SIMD | Yes (lines 236-237) | MEDIUM |
| 350-432 | `matmul_int8_avx2`: AVX2 INT8 GEMM | SIMD | Yes (lines 346-347) | MEDIUM |
| 437-445 | `horizontal_sum_epi32`: Helper for horizontal i32 sum | SIMD | No | LOW |

**Findings for int8_avx2.rs**:

- **F-D5-009 (MEDIUM)**: `debug_assert_eq!` at lines 24, 358-360 for slice length validation. Same release-build elision risk.

- **Positive**: This file has `# Safety` doc comments on all public functions explaining the AVX2 requirement. This is good practice, though the comments don't mention the slice length requirements explicitly.

- The INT8 path uses safe indexing (`input[input_idx]`, `kernel[k_idx]`) within the scalar remainder loops, which will panic on OOB rather than silently reading garbage. This is correct.

---

#### FILE 7: `crates/ruvector-cnn/src/kernels/int8_neon.rs` (9 unsafe refs, 486 lines)

Same structure as int8_avx2.rs but for ARM NEON. Uses `vmull_s8`, `vpadalq_s16` patterns.

- **Positive**: `# Safety` doc comments present.
- **F-D5-010 (MEDIUM)**: Same `debug_assert_eq!` pattern for length checks.
- **F-D5-011 (MEDIUM)**: In `conv2d_int8_neon` (lines 140-146), `u8` input is converted to `i8` via `vsubq_u8(input_u8, vdupq_n_u8(128))`. This shifts the range from [0,255] to [-128,127] correctly for signed NEON multiply. However, this loses the asymmetric quantization zero-point semantics -- the actual zero-point correction is done in the accumulator init, so this is **functionally correct** but confusing without a comment explaining the two-step approach.

---

#### FILE 8: `crates/ruvector-cnn/src/kernels/int8_wasm.rs` (9 unsafe refs, 441 lines)

Same structure for WASM SIMD128. Uses `v128_load`, `i16x8_extend_low_i8x16` patterns.

- **F-D5-012 (MEDIUM)**: Same `debug_assert_eq!` issue.
- **F-D5-013 (MEDIUM)**: In `conv2d_int8_wasm` (lines 146-149), same `u8`-to-`i8` conversion via `u8x16_sub(input_u8, u8x16_splat(128))`, then reinterpret-cast without explicit `transmute`. Line 163: `let input_i8 = input_shifted;` -- this relies on `v128` being a type-erased 128-bit value where `u8x16` operations and `i8x16` operations share the same register, which is correct for WASM SIMD but should have a comment.

---

#### FILE 9: `crates/ruvector-cnn/src/int8/kernels/simd.rs` (5 unsafe refs, 147 lines)

This is an **intermediate stub** that delegates to scalar. Lines 19 and 40: the functions are marked `unsafe` with `#[target_feature(enable = "avx2")]` but simply call the scalar implementation. Line 64: a private `dot_product_int8_avx2` is partially implemented.

- **F-D5-014 (LOW)**: The stub at line 64 processes only the low 128 bits of two 256-bit loads (lines 79-80: `_mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_vec))`), then increments by 16 but the loop says "Process 32 elements". The increment at line 93 is `i += 16` which is correct but the comment is misleading. This is dead code (`#[allow(dead_code)]`), so no production impact.

---

#### FILE 10: `crates/ruvector-cnn/src/simd/winograd.rs` (4 unsafe refs, 482 lines)

| Line(s) | What it does | Category | SAFETY comment? | Risk |
|---------|-------------|----------|-----------------|------|
| 377-386 | `transform_input_avx2`: AVX2 batch input transform (just calls scalar) | SIMD stub | No | LOW |
| 391-399 | `transform_output_avx2`: AVX2 batch output transform (just calls scalar) | SIMD stub | No | LOW |

**Finding**: These are stub implementations that don't use any AVX2 intrinsics despite the name. They simply loop over tiles calling the scalar `transform_input`/`transform_output`. The `unsafe` annotation and `#[target_feature(enable = "avx2")]` are technically unnecessary since no SIMD is used. Not a safety issue, but misleading.

---

#### FILE 11: `crates/ruvector-cnn/src/layers/quantized_conv2d.rs` (2 unsafe refs, 379 lines)

| Line(s) | What it does | Category | SAFETY comment? | Risk |
|---------|-------------|----------|-----------------|------|
| 177-188 | Dispatches to `self.conv_3x3_int8_avx2()` inside `unsafe {}` | Dispatch | No | LOW |
| 286-298 | `conv_3x3_int8_avx2`: `unsafe` method that just calls scalar | SIMD stub | No | LOW |

**Finding**: The AVX2 path (line 297) simply delegates to `self.conv_3x3_int8_scalar(...)`, so the `unsafe` annotation and `#[target_feature]` are placeholder. No actual unsafe operations occur.

---

#### FILE 12: `crates/ruvector-gnn/src/mmap.rs` (11 unsafe refs, 940 lines)

| Line(s) | What it does | Category | SAFETY comment? | Risk |
|---------|-------------|----------|-----------------|------|
| 165-169 | `MmapOptions::new().map_mut(&file)` -- creates mutable mmap | FFI (mmap) | No | MEDIUM |
| 249-252 | `get_embedding`: pointer arithmetic on mmap, `from_raw_parts` | Raw pointer | Yes (line 248) | MEDIUM |
| 294-297 | `set_embedding`: `copy_nonoverlapping` into mmap | Raw pointer | Yes (line 293) | MEDIUM |
| 345-352 | `prefetch`: `libc::madvise` call (Linux only) | FFI (libc) | No | LOW |
| 425-429 | `MmapGradientAccumulator::new`: creates mmap | FFI (mmap) | No | MEDIUM |
| 497-510 | `accumulate`: writes to mmap through `UnsafeCell` | Raw pointer | Yes (line 496) | MEDIUM |
| 570-578 | `get_grad`: reads from mmap through `UnsafeCell` | Raw pointer | Yes (line 569) | MEDIUM |
| 545-549 | `zero_grad`: zeroes mmap through `UnsafeCell` | Raw pointer | No | LOW |
| 602-606 | `Drop` for `MmapGradientAccumulator`: flush through `UnsafeCell` | Raw pointer | No | LOW |
| 612-613 | `unsafe impl Send/Sync for MmapGradientAccumulator` | Manual trait impl | Yes (line 610) | MEDIUM |

**Findings for mmap.rs**:

- **F-D5-015 (MEDIUM)**: The `unsafe impl Send + Sync` at lines 612-613 is justified by the RwLock-based synchronization. The comment at line 610 states "access is protected by RwLocks", which is correct. However, the `UnsafeCell<MmapMut>` pattern means that if any code path reads the mmap without holding a lock, it would be a data race. The current code always holds locks for `accumulate` and `get_grad`, so this is safe in the current implementation.

- **Positive**: All pointer arithmetic uses `checked_mul` and `checked_add` (lines 200-202, 464-470). Bounds are validated with `assert!` (not `debug_assert!`) at lines 223-228, 265-287, 499-501, 573-576. This is a **significant improvement** over the SIMD code and correctly prevents integer overflow attacks.

- **Positive**: The `embedding_offset` method (line 199) returns `Option<usize>` using checked arithmetic, which is the right pattern for untrusted `node_id` inputs.

---

#### FILE 13: `crates/ruvector-gnn/src/cold_tier.rs` (1 unsafe ref)

| Line | What it does | Category | SAFETY comment? | Risk |
|------|-------------|----------|-----------------|------|
| 122-124 | `std::slice::from_raw_parts(features.as_ptr() as *const u8, features.len() * F32_SIZE)` | Raw pointer cast | No | MEDIUM |

**Finding F-D5-016 (MEDIUM)**: Casts `&[f32]` to `&[u8]` for byte-level I/O. This is safe as long as (a) the platform's f32 representation matches IEEE 754 (guaranteed by Rust), and (b) the byte count calculation doesn't overflow. `features.len() * F32_SIZE` could overflow if `features.len() > usize::MAX / 4`, but this is unlikely in practice. Should use `bytemuck::cast_slice` or `std::slice::align_to` instead for clarity and safety.

---

#### FILE 14: `crates/ruvector-gnn/src/lib.rs` (1 unsafe ref)

Line 47: `#![deny(unsafe_op_in_unsafe_fn)]`

This is a **lint directive**, not actual unsafe code. It requires that unsafe operations inside `unsafe fn` be wrapped in explicit `unsafe {}` blocks. This is excellent practice and should be adopted across all D5 crates.

---

#### FILE 15: `crates/sona/src/lora.rs` (1 unsafe ref)

| Line(s) | What it does | Category | SAFETY comment? | Risk |
|---------|-------------|----------|-----------------|------|
| 111-177 | `MicroLoRA::forward_simd`: AVX2 LoRA forward pass using `_mm256_loadu_ps`, `_mm256_fmadd_ps`, `_mm256_storeu_ps` | SIMD (AVX2) | No | MEDIUM |

**Findings for lora.rs**:

- **F-D5-017 (MEDIUM)**: The function is gated behind `#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]` at compile time, which means it only compiles when AVX2 is the target feature. This is safe but means the function is never used via runtime detection. The `forward` method (line 180-188) uses `#[cfg]` to select, not `is_x86_feature_detected!`, so on an x86_64 binary compiled without `-C target-feature=+avx2`, the SIMD path is simply unavailable.

- The unsafe block uses `_mm256_loadu_ps(input[i..].as_ptr())` which gets the pointer from a bounds-checked slice subexpression. The `while i + 8 <= self.hidden_dim` loop at line 127 correctly ensures 8 elements are available. This is **well-bounded**.

---

### 1.3 Pattern Analysis: D1-equivalent vulnerabilities in D5

| D1 Pattern | Present in D5? | Instances | Details |
|-----------|----------------|-----------|---------|
| NEON `debug_assert_eq!` instead of `assert_eq!` | **YES** | 40+ | Every SIMD file uses `debug_assert` for length checks |
| u8 accumulator overflow in SIMD paths | **No** | 0 | INT8 paths correctly use i16->i32 widening (e.g., `_mm256_maddubs_epi16` + `_mm256_madd_epi16`) |
| Integer overflow in capacity calculations | **Partially** | 1 | `cold_tier.rs` line 122 (`features.len() * F32_SIZE`); mmap.rs uses checked arithmetic |
| `transmute` between different-sized types | **No** | 0 | Both transmutes in `quantize.rs` are between same-sized types (32 bytes) |

**New D5-specific patterns not seen in D1**:

| Pattern | Instances | Risk |
|---------|-----------|------|
| `get_unchecked` inside SIMD conv hot paths | 24 in avx2.rs, 2 in neon.rs | HIGH |
| `transmute` for SIMD lane extraction | 2 in quantize.rs | LOW |
| `UnsafeCell<MmapMut>` with manual Send/Sync | 1 in mmap.rs | MEDIUM |
| `libc::madvise` FFI call | 1 in mmap.rs | LOW |
| `from_raw_parts` on mmap pointers | 4 in mmap.rs | MEDIUM |

---

## Part 2: Performance Audit

### 2.1 Benchmark Coverage

| Crate | Benchmarks? | Framework | Coverage |
|-------|------------|-----------|----------|
| ruvector-cnn | **Yes** | Criterion | `cnn_benchmarks.rs` (SIMD ops, layers, activations, pooling) + `int8_bench.rs` (INT8 conv, matmul, quant/dequant, memory) |
| ruvector-attention | **Yes** | Criterion | `attention_bench.rs` (scaled dot product, flash, linear, hyperbolic, MoE, graph, training) + `attention_benchmarks.rs` |
| ruvector-gnn | **No** | N/A | **Missing entirely** -- no benchmarks for forward pass, message passing, or mmap I/O |
| sona | **Yes** | Criterion | `sona_bench.rs` (LoRA forward, learning cycles) |
| neural-trader-* | **No** | N/A | No benchmarks |

**F-D5-PERF-001**: ruvector-gnn has **zero benchmarks**. This is the crate with mmap-based embedding management and gradient accumulation -- exactly the kind of I/O-bound operations that need benchmarking. Missing benchmarks for:
- `MmapManager::get_embedding` / `set_embedding` latency
- `MmapGradientAccumulator::accumulate` under contention
- `FeatureStorage` block-aligned I/O throughput
- GNN layer forward pass
- Differentiable search

### 2.2 Hot Path Analysis

#### ruvector-cnn: Convolution Loops

The convolution hot paths (`conv_3x3_avx2_fma`, `conv_3x3_avx2`) dominate CNN inference time.

**Current optimization level**: Good.
- 4x input channel unrolling with 4 independent FMA accumulators (ILP)
- Tree reduction for accumulator combining (sum01/sum23/sum)
- 8 output channels processed per AVX2 iteration

**F-D5-PERF-002**: Kernel weight gathering is **not vectorized**. In the inner loop (avx2.rs lines 263-282), kernel weights are gathered into a local `[f32; 8]` array element-by-element before loading into a SIMD register:
```rust
for i in 0..8 {
    kv0[i] = *kernel.get_unchecked((oc_idx * in_c + ic_base) * 9 + kernel_offset);
}
let kernel_v0 = _mm256_loadu_ps(kv0.as_ptr());
```
This is a **scatter-gather** pattern that could benefit from `_mm256_i32gather_ps` (AVX2 gather instruction) if the indices fit the stride pattern. However, the kernel layout is `[out_c, in_c, 3, 3]` and the gather stride is `in_c * 9`, which is not constant across the 8 output channels. So the current scalar gather is the pragmatic choice.

**F-D5-PERF-003**: The depthwise convolution (`depthwise_conv_3x3_avx2`, lines 508-623) pre-loads kernel weights into a 3x3 `__m256` cache before the spatial loop. This is good for cache locality. However, it unrolls the 3 kernel rows into separate `if` checks (lines 547, 564, 580), which introduces 3 branches per output pixel. For stride=1 with padding, the middle rows are almost always valid -- only edges need the branch. Consider pre-computing valid row ranges.

#### ruvector-cnn: INT8 Quantization Paths

**F-D5-PERF-004**: The INT8 convolution kernels (`int8_avx2.rs`, `int8_neon.rs`) process input channels in groups of 32 (AVX2) or 16 (NEON), but the inner loop still gathers weights into a local buffer one at a time:
```rust
let mut w_buf = [0i8; 32];
for j in 0..32 {
    w_buf[j] = kernel[k_idx]; // sequential access
}
```
This is O(in_c) scalar operations inside the innermost conv loop. For the INT8 path, the weights should be **pre-packed** in a layout that allows direct SIMD loads (e.g., `[out_c_block, in_c_block, kh, kw, 32]`).

**F-D5-PERF-005**: The Winograd implementation in `winograd.rs` has two AVX2 "stubs" (`transform_input_avx2`, `transform_output_avx2`) that simply call the scalar implementation. The scalar Winograd transform uses 4x4 matrix multiplications that are natural candidates for SIMD vectorization. The current implementation uses triple-nested scalar loops (lines 112-133, 154-175, 196-216), each doing a 4x3 or 4x4 matmul. These could be vectorized with 4-wide SIMD (either AVX2 or NEON).

#### ruvector-attention: Forward Pass / Softmax

Ruvector-attention contains **zero unsafe code** and relies on pure Rust for all computation. This is excellent for safety but may leave performance on the table.

**F-D5-PERF-006**: The attention crate has 69 source files covering scaled dot product, flash attention, linear attention, hyperbolic attention, MoE routing, and more. All use safe Rust with `Vec<f32>` operations. The softmax and dot product computations in these paths could benefit from SIMD dispatch similar to ruvector-cnn's pattern, especially for the `FlashAttention` block-tiled implementation.

#### ruvector-gnn: Message Passing / Aggregation

The GNN crate's core operations are in `layer.rs`, `search.rs`, and `training.rs`. These use standard Rust iterators and Vec operations.

**F-D5-PERF-007**: The `MmapManager` allocates a `Vec<f32>` for every `apply` call in the gradient accumulator (mmap.rs line 530: `let mut updated = vec![0.0f32; self.d_embed]`). For large-scale training with millions of nodes, this creates millions of heap allocations per gradient step. A pre-allocated update buffer should be reused.

### 2.3 Memory Allocation Patterns

| Location | Pattern | Issue | Severity |
|----------|---------|-------|----------|
| `avx2.rs:156-157` | `vec![0.0; channels]` x2 inside `batch_norm_avx2` | Per-call alloc in hot path | MEDIUM |
| `neon.rs:158-159` | `vec![0.0; channels]` x2 inside `batch_norm_neon` | Per-call alloc in hot path | MEDIUM |
| `wasm.rs:124-125` | `vec![0.0; channels]` x2 inside `batch_norm_wasm` | Per-call alloc in hot path | MEDIUM |
| `winograd.rs:315` | `vec![0.0; out_c * 4]` per 2x2 output tile | Per-tile alloc in convolution | HIGH |
| `mmap.rs:530` | `vec![0.0; d_embed]` per node in `apply` | Per-node alloc in gradient step | HIGH |
| `int8_avx2.rs:105-117` | `vec![0i32; out_c]` for weight_sums | Per-call alloc, acceptable | LOW |

**F-D5-PERF-008**: The Winograd implementation allocates a `Vec` on every 2x2 output tile (line 315). For a 224x224 image with 64 output channels, this is `112 * 112 = 12,544` heap allocations per layer. This should be hoisted outside the tile loop with a single allocation.

### 2.4 Algorithmic Efficiency

| Operation | Current | Optimal | Gap |
|-----------|---------|---------|-----|
| 3x3 FP32 convolution | SIMD (AVX2/NEON/WASM) with 4x unrolling | im2col + GEMM or Winograd | Winograd available but stubs not SIMD-ized |
| INT8 convolution | SIMD with scalar weight gather | Pre-packed weights + direct SIMD load | 2-3x improvement possible |
| Dot product | SIMD with 4x unrolling (NEON), 8x (AVX2) | Already good | Minimal gap |
| Batch normalization | SIMD with pre-computed scale/shift | Fused with activation (BN-ReLU fusion) | Not fused |
| Global average pooling | SIMD accumulation | Already good | Minimal gap |
| GNN message passing | Safe Rust iterators | SIMD-accelerated or parallel | No SIMD path exists |

**F-D5-PERF-009**: Batch normalization is not fused with the following activation layer. A fused BN-ReLU or BN-ReLU6 kernel would eliminate one pass over the tensor data, saving memory bandwidth. This is standard in production CNN frameworks.

### 2.5 Cache Locality

The CNN code uses **NHWC** (channels-last) layout throughout, which is correct for:
- SIMD processing of multiple channels simultaneously
- AVX2 8-wide and NEON 4-wide channel-parallel operations

However, the INT8 convolution's weight layout is `[out_c, in_c, kh, kw]` (OIHW), which creates non-contiguous access patterns when gathering weights for 8 output channels. A packed layout like `[out_c/8, in_c/32, kh, kw, 32, 8]` would enable direct SIMD loads.

---

## Part 3: NAPI/WASM Bindings Audit

### 3.1 ruvector-cnn-wasm (470 lines)

**No unsafe code**. The WASM bindings use `wasm_bindgen` with safe Rust wrappers. Data marshaling is through `Vec<f32>` and `Vec<u8>` copies. No memory leak risk.

Input validation is present (line 72-77): checks `image_data.len() == width * height * 3` before processing. Good.

### 3.2 ruvector-gnn-node (421 lines)

**No unsafe code**. NAPI-RS bindings with safe wrappers. `Float32Array` data is copied to/from `Vec<f32>` (line 85: `arr.to_vec()`). No ownership transfer issues.

### 3.3 ruvector-attention-node (5 files)

**No unsafe code**. Safe NAPI-RS bindings.

### 3.4 FFI Memory Safety Summary

All D5 NAPI/WASM bindings are **safe**. They copy data at the FFI boundary rather than sharing pointers, which eliminates use-after-free and double-free risks at the cost of copy overhead. For large tensors, this copy overhead could be significant, but it is the correct trade-off for safety.

---

## Consolidated Findings

### Critical / High Severity

| ID | File | Line(s) | Finding | Category | Risk |
|----|------|---------|---------|----------|------|
| F-D5-001 | avx2.rs | 254-310, 408-462 | `get_unchecked` in conv hot paths with no release-build bounds checks on caller inputs | Buffer overread | HIGH |
| F-D5-002 | All SIMD files | Multiple | 40+ `debug_assert_eq!` for slice length validation, stripped in release | Missing bounds checks | HIGH |
| F-D5-005 | neon.rs | 79 | `get_unchecked` in dot product remainder loop; `debug_assert_eq` at line 19 is the only guard | Buffer overread | HIGH |
| F-D5-008 | quantize.rs | 361, 394 | `std::mem::transmute` between `__m256i` and `[i32; 8]`; correct but avoidable | Transmute | HIGH |
| F-D5-PERF-001 | ruvector-gnn | N/A | Zero benchmarks for the entire crate including mmap I/O, GNN layer forward, differentiable search | Missing benchmarks | HIGH |

### Medium Severity

| ID | File | Line(s) | Finding | Risk |
|----|------|---------|---------|------|
| F-D5-004 | mod.rs | All dispatch fns | Dispatch functions call unsafe impls without enforcing preconditions | MEDIUM |
| F-D5-006 | neon.rs | 158-159 | Heap allocation inside unsafe fn hot path | MEDIUM |
| F-D5-007 | wasm.rs | 17,55,80,120 | `debug_assert` only for length checks | MEDIUM |
| F-D5-009 | int8_avx2.rs | 24,358-360 | `debug_assert_eq!` for length validation | MEDIUM |
| F-D5-010 | int8_neon.rs | 23,341-343 | `debug_assert_eq!` for length validation | MEDIUM |
| F-D5-012 | int8_wasm.rs | 22,312-314 | `debug_assert_eq!` for length validation | MEDIUM |
| F-D5-015 | mmap.rs | 612-613 | `unsafe impl Send+Sync` for `MmapGradientAccumulator` | MEDIUM |
| F-D5-016 | cold_tier.rs | 122 | `from_raw_parts` f32-to-u8 cast without overflow check on length | MEDIUM |

### Low Severity

| ID | File | Line(s) | Finding | Risk |
|----|------|---------|---------|------|
| F-D5-003 | avx2.rs | 156-157 | Heap alloc in hot path (batch_norm) | LOW |
| F-D5-014 | int8/simd.rs | 64-103 | Dead code with misleading comment | LOW |
| F-D5-017 | lora.rs | 111-177 | Compile-time SIMD gate (cfg) instead of runtime detection | LOW |
| PERF-005 | winograd.rs | 377-399 | AVX2 "stubs" that just call scalar | LOW |

### Performance Findings

| ID | Finding | Impact | Effort |
|----|---------|--------|--------|
| F-D5-PERF-002 | Kernel weight gathering not vectorized in conv hot path | Low (scatter pattern) | Medium |
| F-D5-PERF-003 | Depthwise conv branches per-pixel for valid row check | Low | Low |
| F-D5-PERF-004 | INT8 weights not pre-packed for SIMD-aligned loads | 2-3x improvement potential | High |
| F-D5-PERF-005 | Winograd AVX2 stubs are scalar | Winograd benefit lost on x86 | Medium |
| F-D5-PERF-006 | ruvector-attention has no SIMD acceleration | Unknown until benchmarked | High |
| F-D5-PERF-007 | Per-node heap alloc in gradient accumulator apply | High for large graphs | Low |
| F-D5-PERF-008 | Per-tile heap alloc in Winograd convolution | High (12K allocs/layer) | Low |
| F-D5-PERF-009 | Batch norm not fused with activation | ~2x memory bandwidth | Medium |

---

## Recommendations (Priority Order)

### P0: Must-Fix (Security / Correctness)

1. **Replace all `debug_assert_eq!` with `assert_eq!` in SIMD dispatch functions** (`mod.rs` lines 35, 71, etc.). These are the public API entry points and must validate inputs in release builds. The per-element overhead of a single length check is negligible compared to the actual SIMD computation.

2. **Add `assert!` bounds checks at entry to all `get_unchecked` functions** (`conv_3x3_avx2_fma`, `conv_3x3_avx2`, `depthwise_conv_3x3_avx2`). At minimum:
   ```rust
   assert!(input.len() >= in_h * in_w * in_c);
   assert!(kernel.len() >= out_c * in_c * 9);
   assert!(output.len() >= out_h * out_w * out_c);
   ```

3. **Replace `transmute` in quantize.rs** with `_mm256_storeu_si256` / `_mm256_loadu_si256` alternatives.

4. **Adopt `#![deny(unsafe_op_in_unsafe_fn)]`** across all D5 crates. Currently only ruvector-gnn has this.

### P1: Should-Fix (Performance)

5. **Add SAFETY comments** to all `unsafe` blocks in SIMD code. Currently avx2.rs, neon.rs, wasm.rs, and quantize.rs have zero SAFETY comments.

6. **Hoist heap allocations out of hot paths**: batch_norm scale/shift vectors, Winograd tile_output vector, and gradient accumulator update buffer should be pre-allocated and reused.

7. **Add benchmarks for ruvector-gnn**: MmapManager, GradientAccumulator, GNN layer forward, differentiable search.

### P2: Nice-to-Have (Optimization)

8. **Pre-pack INT8 weights** for SIMD-aligned loading in the conv kernels.
9. **Implement fused BN-ReLU/BN-ReLU6 kernels**.
10. **Vectorize Winograd transform stubs** with actual AVX2/NEON implementations.
11. **Consider SIMD dispatch** for ruvector-attention hot paths (softmax, dot product).
12. **Use `bytemuck::cast_slice`** in cold_tier.rs instead of manual `from_raw_parts`.

---

## Security Score

**Weighted Finding Score**: 3 x HIGH(5) + 2 x MEDIUM(8) + 1 x LOW(4) = 15 + 16 + 4 = **35 / 100** (points deducted)

**Security Score**: 65/100 -- **CONDITIONAL PASS**

The unsafe code is concentrated in well-defined SIMD paths with correct algorithmic logic, but the systematic absence of release-build bounds checking creates a class of potential buffer overread vulnerabilities. The mmap code in ruvector-gnn is notably better-defended with checked arithmetic and runtime assertions.

**Recommendation**: CONDITIONAL MERGE -- P0 items (bounds checks in dispatch layer, transmute replacement) should be addressed before the next release. No evidence of exploitable vulnerabilities in current calling patterns, but the defensive surface is insufficient for a security-critical domain.

---

## Files Examined

### ruvector-cnn (src/)
- `simd/avx2.rs` (814 lines)
- `simd/mod.rs` (387 lines)
- `simd/neon.rs` (534 lines)
- `simd/wasm.rs` (551 lines)
- `simd/quantize.rs` (559 lines)
- `simd/winograd.rs` (482 lines)
- `kernels/int8_avx2.rs` (587 lines)
- `kernels/int8_neon.rs` (486 lines)
- `kernels/int8_wasm.rs` (441 lines)
- `int8/kernels/simd.rs` (147 lines)
- `layers/quantized_conv2d.rs` (379 lines)

### ruvector-cnn (benches/)
- `cnn_benchmarks.rs`
- `int8_bench.rs` (387 lines)

### ruvector-gnn (src/)
- `mmap.rs` (940 lines)
- `cold_tier.rs` (first 135 lines + line 122 context)
- `lib.rs` (98 lines)

### ruvector-attention (benches/)
- `attention_bench.rs` (first 80 lines)

### sona (src/)
- `lora.rs` (519 lines)

### FFI Bindings
- `ruvector-cnn-wasm/src/lib.rs` (470 lines)
- `ruvector-gnn-node/src/lib.rs` (421 lines)
- `ruvector-attention-node/src/` (5 files)
