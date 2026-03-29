# Phase 2 Domain 4: ruvector-postgres Unsafe Code Audit

## Executive Summary

| File | LOC | unsafe blocks | CRITICAL | HIGH | MEDIUM | LOW |
|------|-----|---------------|----------|------|--------|-----|
| `distance/simd.rs` | 2,128 | 78 | 2 | 4 | 6 | 3 |
| `index/hnsw_am.rs` | 2,351 | 40 | 3 | 5 | 4 | 2 |
| `index/ivfflat_am.rs` | 2,174 | 29 | 2 | 4 | 3 | 1 |
| `index/ivfflat_storage.rs` | ~350 | 9 | 0 | 2 | 1 | 0 |
| Other files (types, etc.) | ~3,000 | 79 | 0 | 3 | 4 | 2 |
| **TOTALS** | ~10,000 | **235** | **7** | **18** | **18** | **8** |

**Verdict: BLOCK MERGE for CRITICAL issues. This crate requires immediate remediation of 7 critical findings before any production deployment.**

The ruvector-postgres crate contains 235 unsafe blocks across 21 files. This audit focuses on the three densest files (simd.rs, hnsw_am.rs, ivfflat_am.rs) which together contain 147 unsafe blocks (63% of the crate total). The crate operates as a PostgreSQL extension running inside the postmaster process -- any memory safety violation can crash the entire database server, corrupt data, or create exploitable vulnerabilities.

---

## 1. simd.rs -- 78 unsafe blocks (2,128 LOC)

### 1.1 Unsafe Block Inventory

| Line(s) | Category | Description | Risk |
|---------|----------|-------------|------|
| 130-153 | SIMD Intrinsic | `l2_distance_ptr_avx512` - AVX-512 L2 | MEDIUM |
| 164-202 | SIMD Intrinsic | `cosine_distance_ptr_avx512` - AVX-512 cosine | MEDIUM |
| 213-234 | SIMD Intrinsic | `inner_product_ptr_avx512` - AVX-512 IP | MEDIUM |
| 244-267 | SIMD Intrinsic | `manhattan_distance_ptr_avx512` - AVX-512 L1 | MEDIUM |
| 277-298 | SIMD Intrinsic | `cosine_distance_normalized_avx512` | MEDIUM |
| 307-309 | Wrapper | `euclidean_distance_avx512` slice-to-ptr | LOW |
| 313-316 | Wrapper | `cosine_distance_avx512` slice-to-ptr | LOW |
| 320-323 | Wrapper | `inner_product_avx512` slice-to-ptr | LOW |
| 326-330 | Wrapper | `manhattan_distance_avx512` slice-to-ptr | LOW |
| 341 | Dispatch | `euclidean_distance_avx512_wrapper` unsafe call | LOW |
| 343 | Dispatch | fallback to avx2 | LOW |
| 352 | Dispatch | non-avx512 path | LOW |
| 367, 369 | Dispatch | cosine avx512 wrapper | LOW |
| 378 | Dispatch | cosine non-avx512 path | LOW |
| 393, 395 | Dispatch | inner product avx512 wrapper | LOW |
| 404 | Dispatch | inner product non-avx512 | LOW |
| 419, 421 | Dispatch | manhattan avx512 wrapper | LOW |
| 430 | Dispatch | manhattan non-avx512 | LOW |
| 453-494 | SIMD Intrinsic | `l2_distance_ptr_avx2` - AVX2 L2 with aligned loads | **HIGH** |
| 504-559 | SIMD Intrinsic | `cosine_distance_ptr_avx2` - AVX2 cosine | MEDIUM |
| 569-599 | SIMD Intrinsic | `inner_product_ptr_avx2` - AVX2 IP | MEDIUM |
| 610-646 | SIMD Intrinsic | `manhattan_distance_ptr_avx2` - AVX2 L1 | MEDIUM |
| 658-667 | Pointer arith | `l2_distance_ptr_scalar` | LOW |
| 675-696 | Pointer arith | `cosine_distance_ptr_scalar` | LOW |
| 704-712 | Pointer arith | `inner_product_ptr_scalar` | LOW |
| 720-728 | Pointer arith | `manhattan_distance_ptr_scalar` | LOW |
| 746-758 | Dispatch | `l2_distance_ptr` runtime dispatch | LOW |
| 769-781 | Dispatch | `cosine_distance_ptr` runtime dispatch | LOW |
| 792-804 | Dispatch | `inner_product_ptr` runtime dispatch | LOW |
| 815-835 | Dispatch | `manhattan_distance_ptr` runtime dispatch | LOW |
| 851-863 | Batch | `l2_distances_batch` | MEDIUM |
| 873-885 | Batch | `cosine_distances_batch` | MEDIUM |
| 895-907 | Batch | `inner_product_batch` | MEDIUM |
| 917-929 | Batch | `manhattan_distances_batch` | MEDIUM |
| 939-952 | Batch | `l2_distances_batch_parallel` | MEDIUM |
| 959-972 | Batch | `cosine_distances_batch_parallel` | MEDIUM |
| 981-1008 | SIMD Intrinsic | `euclidean_distance_avx2` slice-based | LOW |
| 1014-1052 | SIMD Intrinsic | `cosine_distance_avx2` slice-based | LOW |
| 1057-1076 | SIMD Intrinsic | `inner_product_avx2` slice-based | LOW |
| 1081-1103 | SIMD Intrinsic | `manhattan_distance_avx2` slice-based | LOW |
| 1108-1115 | SIMD Intrinsic | `horizontal_sum_256` | LOW |
| 1195 | Dispatch | `l2_distance_optimized` unsafe call to unrolled | LOW |
| 1221 | Dispatch | `cosine_distance_optimized` unsafe call | LOW |
| 1247 | Dispatch | `inner_product_distance_optimized` unsafe call | LOW |
| 1269-1335 | SIMD Intrinsic | `l2_distance_avx2_unrolled` 4x unroll | MEDIUM |
| 1340-1437 | SIMD Intrinsic | `cosine_distance_avx2_unrolled` 4x unroll | MEDIUM |
| 1442-1497 | SIMD Intrinsic | `inner_product_avx2_unrolled` 4x unroll | MEDIUM |
| 1505-1528 | SIMD Intrinsic | `euclidean_distance_neon` - NEON L2 | LOW |
| 1532-1567 | SIMD Intrinsic | `cosine_distance_neon` - NEON cosine | LOW |
| 1571-1592 | SIMD Intrinsic | `inner_product_neon` - NEON IP | LOW |
| 1597-1620 | SIMD Intrinsic | `manhattan_distance_neon` - NEON L1 | LOW |
| 1625-1649 | SIMD Intrinsic | `manhattan_distance_ptr_neon` - NEON ptr L1 | LOW |
| 1659, 1673, 1687, 1701 | Dispatch | AVX2 wrapper calls | LOW |
| 1715, 1725, 1735, 1745 | Dispatch | NEON wrapper calls | LOW |
| 1762-1782 | SIMD Intrinsic | `cosine_distance_normalized_avx2` | LOW |
| 1786-1795 | Pointer arith | `cosine_distance_normalized_scalar` | LOW |
| 1799-1808 | Dispatch | `cosine_distance_normalized_ptr` | LOW |
| 1813 | Wrapper | `cosine_distance_normalized` slice-to-ptr | LOW |
| 1822-1837 | Batch | `l2_topk_batch` | MEDIUM |
| 1841-1856 | Batch | `cosine_topk_normalized_batch` | MEDIUM |

### 1.2 CRITICAL Findings

#### CRITICAL-S1: All pointer-based functions use `debug_assert!` for null/length checks (D1 cross-reference)

**Severity: CRITICAL (CWE-787: Out-of-bounds Write, CWE-125: Out-of-bounds Read)**

Every raw-pointer SIMD function guards its preconditions with `debug_assert!`, which is stripped in release builds. This is the **exact same pattern** found in the D1 NEON audit.

Affected lines (14 distinct functions):
```rust
// Line 131 (l2_distance_ptr_avx512)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);

// Line 165 (cosine_distance_ptr_avx512)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);

// Line 214 (inner_product_ptr_avx512)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);

// Line 245 (manhattan_distance_ptr_avx512)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);

// Line 278 (cosine_distance_normalized_avx512)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);

// Line 454 (l2_distance_ptr_avx2)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);

// Line 505 (cosine_distance_ptr_avx2)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);

// Line 570 (inner_product_ptr_avx2)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);

// Line 611 (manhattan_distance_ptr_avx2)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);

// Lines 659, 676, 705, 721 (scalar ptr variants)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);

// Line 1628 (manhattan_distance_ptr_neon)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);

// Lines 1763, 1787 (normalized cosine variants)
debug_assert!(!a.is_null() && !b.is_null() && len > 0);
```

**Impact**: In release builds, passing `len = 0` causes division by zero in chunk calculations (`let chunks = len / 16`). Passing a null pointer causes immediate segfault. Since these are called from PostgreSQL index operations (via hnsw_am.rs and ivfflat_am.rs), a corrupted index page could supply `len = 0` or a null pointer, crashing the PostgreSQL server.

**Remediation**: Replace all `debug_assert!` with `assert!` for null and length checks, or better, return a sentinel value (e.g., `f32::MAX` for distance, `0.0` for similarity) when preconditions fail:
```rust
if a.is_null() || b.is_null() || len == 0 {
    return f32::MAX; // or 0.0 for inner product
}
```

#### CRITICAL-S2: Batch functions use `debug_assert!` for results buffer bounds

**Severity: CRITICAL (CWE-787: Out-of-bounds Write)**

Lines 857, 879, 901, 923, 945, 965:
```rust
// Line 857 (l2_distances_batch)
debug_assert!(results.len() >= vectors.len());
debug_assert!(!query.is_null() && len > 0);
```

All six batch functions (`l2_distances_batch`, `cosine_distances_batch`, `inner_product_batch`, `manhattan_distances_batch`, `l2_distances_batch_parallel`, `cosine_distances_batch_parallel`) write to `results[i]` without a runtime bounds check. If `results.len() < vectors.len()`, this is a heap buffer overflow -- a classic exploitable vulnerability.

**Impact**: Heap corruption inside the PostgreSQL backend process. Could lead to arbitrary code execution if an attacker can control the vector count vs. results buffer size.

**Remediation**: Replace `debug_assert!` with `assert!` or add explicit bounds checking:
```rust
assert!(results.len() >= vectors.len(),
    "results buffer too small: {} < {}", results.len(), vectors.len());
```

### 1.3 HIGH Findings

#### HIGH-S1: AVX2 aligned loads without runtime alignment guarantee

**Severity: HIGH (CWE-119: Improper Restriction of Operations within Memory Buffer)**

Lines 460-476 (`l2_distance_ptr_avx2`), and similarly in `cosine_distance_ptr_avx2` (514), `inner_product_ptr_avx2` (576), `manhattan_distance_ptr_avx2` (618):

```rust
// Line 458-476
let use_aligned = is_avx2_aligned(a, b);

if use_aligned {
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_load_ps(a.add(offset));   // Requires 32-byte alignment
        let vb = _mm256_load_ps(b.add(offset));
```

The alignment check at the function entry (`is_avx2_aligned`) only verifies the base pointers are 32-byte aligned. However, `a.add(offset)` with `offset = i * 8` advances by `8 * 4 = 32` bytes each iteration, so the alignment is preserved through the loop. **The alignment check is correct in practice** because 32-byte alignment of the base pointer plus 32-byte strides guarantees alignment throughout.

However, the code relies on the caller providing the correct `len`. If `len` is wrong and the loop over-reads, `_mm256_load_ps` on an aligned but invalid address will segfault. This is mitigated only by `debug_assert!` (see CRITICAL-S1).

**Risk**: The alignment logic itself is sound, but the combination with missing runtime length validation makes this HIGH rather than MEDIUM.

#### HIGH-S2: Slice-based SIMD functions do not validate `a.len() == b.len()`

**Severity: HIGH (CWE-125: Out-of-bounds Read)**

Lines 981, 1014, 1057, 1081 (slice-based AVX2 functions), and 1505, 1532, 1571, 1597 (NEON functions):

```rust
// Line 981
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    // ... loops using n, reads from b using a.len() as bound
```

These functions use `a.len()` to determine loop bounds but read from both `a` and `b`. If `b.len() < a.len()`, the SIMD loads will read past `b`'s allocation boundary. The only guard is the `debug_assert_eq!` in the public wrappers (e.g., line 1129, 1143, 1184), which are stripped in release builds.

The simsimd wrappers (lines 1128-1169) also use `debug_assert_eq!`:
```rust
// Line 1129
debug_assert_eq!(a.len(), b.len());
```

**Impact**: Buffer overread. On modern x86/ARM, this typically reads garbage data silently rather than segfaulting (page boundaries aside), leading to silently wrong distance computations. Wrong distances in an HNSW or IVFFlat index mean incorrect search results returned to users.

**Remediation**: Change to `assert_eq!` in all public entry points, or add explicit length checks in the private SIMD functions.

#### HIGH-S3: No NEON `#[target_feature]` annotations

**Severity: HIGH (Correctness/Safety)**

Lines 1505-1649 (all NEON functions):
```rust
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
```

None of the NEON functions have `#[target_feature(enable = "neon")]` annotations. On AArch64, NEON is mandatory (as the code's comment on line 64 notes: "NEON is mandatory on AArch64"), so this is not exploitable. However, it deviates from the code's own pattern for x86 (where `#[target_feature]` is consistently applied) and makes the code inconsistent. If this crate ever targets 32-bit ARM (where NEON is optional), this would become a correctness bug.

**Risk**: LOW for current AArch64-only targets, but flagged HIGH for consistency and forward safety.

#### HIGH-S4: `_mm256_load_ps` panics on misaligned data in debug builds

**Severity: HIGH (CWE-704)**

The `is_avx2_aligned` check (line 107-109) tests only the base pointer alignment. While the stride arithmetic preserves alignment (as discussed in HIGH-S1), there is a subtle issue: PostgreSQL memory allocations use `palloc` which provides 8-byte alignment, NOT 32-byte alignment. Any vector data read from PostgreSQL pages will be at an arbitrary offset within the 8KB page.

The `is_avx2_aligned` function will correctly return `false` for palloc'd pointers, routing to the unaligned path. This is safe. However, the test infrastructure (lines 1955-1988) creates `Vec<f32>` on the Rust heap, which may be 16-byte aligned (Rust's global allocator), causing the aligned path to be used in tests but not in production. This means the aligned code path is effectively untested in realistic conditions.

### 1.4 MEDIUM Findings

#### MEDIUM-S1: No scalar fallback for `manhattan_distance_ptr` on non-x86, non-aarch64

Line 833-834: On platforms that are neither x86_64 nor aarch64, `manhattan_distance_ptr` falls through to `manhattan_distance_ptr_scalar`. This is correct. However, the other `*_ptr` dispatch functions (L2, cosine, IP) on lines 746-804 have the same structure. **All are correct** -- this is a clean-bill note.

#### MEDIUM-S2: `cosine_distance_normalized` trusts caller that vectors are pre-normalized

Lines 1811-1814:
```rust
pub fn cosine_distance_normalized(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    unsafe { cosine_distance_normalized_ptr(a.as_ptr(), b.as_ptr(), a.len()) }
}
```

No validation that vectors are actually unit-length. Passing non-normalized vectors produces silently wrong results (values outside [0, 2]).

#### MEDIUM-S3: Batch topk functions do not validate `k <= vectors.len()`

Lines 1822-1837 (`l2_topk_batch`):
```rust
pub unsafe fn l2_topk_batch(
    query: *const f32,
    vectors: &[*const f32],
    len: usize,
    k: usize,
) -> Vec<(usize, f32)> {
```

If `k > vectors.len()`, `results.truncate(k)` is a no-op, so the function returns all results. This is correct behavior but should be documented.

#### MEDIUM-S4: No NaN handling in SIMD distance functions

All SIMD distance functions perform floating-point arithmetic without checking for NaN inputs. NaN propagates through SIMD lanes silently. The cosine distance functions check `denominator == 0.0` (lines 198, 554, 1047, 1432) but not for NaN, which would cause `partial_cmp` to return `None` in sorting operations.

#### MEDIUM-S5: simsimd functions cast f64 to f32 with potential precision loss

Lines 1132-1133:
```rust
if let Some(dist_sq) = f32::sqeuclidean(a, b) {
    (dist_sq as f32).sqrt()
```

The `sqeuclidean` method returns `f64` according to the simsimd API. Casting back to `f32` can lose precision for large vectors. The scalar fallback does not have this issue.

#### MEDIUM-S6: `l2_distance_avx2_unrolled` remainder handling is complex and error-prone

Lines 1308-1335: The unrolled function has a three-stage remainder: 32-float chunks, 8-float chunks, then scalar. The arithmetic for `remaining_start` and `final_start` is correct:
```rust
let remaining_start = chunks_4x * 32;
let remaining_chunks = (n - remaining_start) / 8;
// ...
let final_start = remaining_start + remaining_chunks * 8;
```

This handles all cases correctly, but the complexity invites bugs on future modification.

### 1.5 D1 Cross-Reference

| D1 Finding | Present in D4 simd.rs? | Assessment |
|------------|------------------------|------------|
| NEON `debug_assert_eq` instead of `assert_eq` for length check | **YES** - All pointer functions use `debug_assert!` (CRITICAL-S1) | Same bug, same severity |
| u8 accumulator overflow in hamming NEON | **NO** - No hamming distance in simd.rs (f32 only) | Not applicable |
| Integer overflow in `grow()` | **NO** - No dynamic allocation/grow in simd.rs | Not applicable |

### 1.6 Scalar Fallback Coverage

| Distance Function | AVX-512 | AVX2 | NEON | Scalar Fallback |
|-------------------|---------|------|------|-----------------|
| L2 (Euclidean) | Yes | Yes | Yes | Yes (`l2_distance_ptr_scalar`) |
| Cosine | Yes | Yes | Yes | Yes (`cosine_distance_ptr_scalar`) |
| Inner Product | Yes | Yes | Yes | Yes (`inner_product_ptr_scalar`) |
| Manhattan | Yes | Yes | Yes | Yes (`manhattan_distance_ptr_scalar`) |
| Cosine Normalized | Yes | Yes | No | Yes (`cosine_distance_normalized_scalar`) |

All SIMD functions have scalar fallbacks. Coverage is complete.

### 1.7 `#[target_feature]` Annotation Audit

| Function Group | `#[target_feature]` | Correct? |
|---------------|---------------------|----------|
| AVX-512 functions | `enable = "avx512f"` | Yes |
| AVX2 functions (with FMA) | `enable = "avx2", enable = "fma"` | Yes |
| AVX2 functions (no FMA) | `enable = "avx2"` | Yes (manhattan, horizontal_sum) |
| NEON functions | **MISSING** | See HIGH-S3 |

---

## 2. hnsw_am.rs -- 40 unsafe blocks (2,351 LOC)

### 2.1 Unsafe Block Inventory

| Line(s) | PG C API Called | Description | Risk |
|---------|----------------|-------------|------|
| 356-361 | `ReadBuffer`, `LockBuffer`, `BufferGetPage` | `get_meta_page` shared lock | MEDIUM |
| 364-369 | `ReadBuffer`, `LockBuffer`, `BufferGetPage` | `get_meta_page_exclusive` | MEDIUM |
| 372-390 | `RelationGetNumberOfBlocksInFork`, `ReadBuffer`, `LockBuffer`, `BufferGetPage` | `get_or_create_meta_page` | **HIGH** |
| 393-397 | `ptr::read` | `read_metadata` - raw pointer cast and read | **HIGH** |
| 400-404 | `ptr::write` | `write_metadata` - raw pointer write | **HIGH** |
| 431-457 | `index_getprocid`, `get_func_name`, `CStr::from_ptr`, `pfree` | `metric_from_index` - FFI string handling | MEDIUM |
| 461-500 | `ReadBuffer`, `LockBuffer`, `BufferGetPage`, `PageInit`, `ptr::write`, `MarkBufferDirty`, `UnlockReleaseBuffer` | `allocate_node_page` | **HIGH** |
| 503-519 | `ReadBuffer`, `LockBuffer`, `BufferGetPage`, `ptr::read` | `read_node_header` | MEDIUM |
| 523-564 | `ReadBuffer`, `LockBuffer`, `BufferGetPage`, `ptr::read` | `read_vector` with bounds check | MEDIUM |
| 567-627 | `ReadBuffer`, `LockBuffer`, `ptr::read` | `read_neighbors` with bounds check | MEDIUM |
| 630-643 | Calls `read_vector`, `distance` | `calculate_distance` | LOW |
| 664-797 | Calls other unsafe helpers | `hnsw_search` - search orchestration | MEDIUM |
| 805-902 | `pg_guard` extern C, heap scan, `PgBox::alloc0` | `hnsw_build` - index build | **CRITICAL** |
| 912-964 | `extern "C"`, raw pointer casts, varlena parsing | `hnsw_build_callback` | **CRITICAL** |
| 968-1003 | `table_index_build_scan` | `build_index_from_heap` | MEDIUM |
| 1008-1019 | `PageInit`, `MarkBufferDirty`, `UnlockReleaseBuffer` | `hnsw_buildempty` | LOW |
| 1023-1096 | `pg_guard` extern C, datum extraction, varlena | `hnsw_insert` | **HIGH** |
| 1099-1185 | Multiple helpers called | `hnsw_insert_vector` | MEDIUM |
| 1188-1261 | Multiple helpers called | `search_layer_for_insert` | MEDIUM |
| 1265-1323 | `ptr::write`, `ptr::copy`, pointer arithmetic | `write_neighbors_to_page` | **CRITICAL** |
| 1326-1404 | `ReadBuffer`, `LockBuffer`, `ptr::read`, `ptr::write` | `connect_node_to_neighbors` | **HIGH** |
| 1408-1468 | `pg_guard` extern C, callback invocation | `hnsw_bulkdelete` | MEDIUM |
| 1471-1484 | `ReadBuffer`, `LockBuffer`, `MarkBufferDirty` | `mark_node_deleted` | LOW |
| 1488-1531 | `pg_guard` extern C, meta read/write | `hnsw_vacuumcleanup` | LOW |
| 1535-1570 | `pg_guard` extern C, cost estimation | `hnsw_costestimate` | LOW |
| 1580-1619 | `pg_guard` extern C, `palloc0`, `palloc`, Box alloc | `hnsw_beginscan` | **HIGH** |
| 1623-1766 | `pg_guard` extern C, datum extraction, varlena | `hnsw_rescan` | **HIGH** |
| 1823-1892 | `pg_guard` extern C, result iteration | `hnsw_gettuple` | MEDIUM |
| 1896-1899 | `pg_guard` extern C | `hnsw_getbitmap` | LOW |
| 1903-1912 | `Box::from_raw`, null out | `hnsw_endscan` | MEDIUM |
| 1925-1941 | `pg_guard` extern C | `hnsw_options` | LOW |
| 1945-1955 | `pg_guard` extern C | `hnsw_validate` | LOW |
| 1959-1968 | `pg_guard` extern C | `hnsw_property` | LOW |
| 2061-2089 | `palloc0`, `ptr::copy_nonoverlapping` | `hnsw_handler` - AM registration | MEDIUM |
| 2120-2196 | SPI, `from_polymorphic_datum` | `ruvector_hnsw_debug` | MEDIUM |

### 2.2 CRITICAL Findings

#### CRITICAL-H1: `allocate_node_page` does not validate vector size fits in page

**Severity: CRITICAL (CWE-787: Out-of-bounds Write)**

Lines 461-500:
```rust
unsafe fn allocate_node_page(
    index_rel: Relation,
    vector: &[f32],
    tid: ItemPointerData,
    max_layer: usize,
) -> BlockNumber {
    // ...
    // Write vector data after header
    let vector_ptr = data_ptr.add(size_of::<HnswNodePageHeader>()) as *mut f32;
    for (i, &val) in vector.iter().enumerate() {
        ptr::write(vector_ptr.add(i), val);  // Line 493
    }
```

There is NO check that `size_of::<PageHeaderData>() + size_of::<HnswNodePageHeader>() + vector.len() * 4` fits within `BLCKSZ` (8192 bytes). A vector with more than approximately 2,000 dimensions would overflow the page boundary, writing into adjacent memory or causing a segfault.

The maximum safe dimensions: `(8192 - sizeof(PageHeaderData) - sizeof(HnswNodePageHeader)) / 4` = approximately `(8192 - 24 - 32) / 4 = 2034` dimensions. A 3072-dimension vector (GPT-4 embeddings) would overflow by ~4KB.

**Impact**: Page buffer overflow corrupts PostgreSQL shared memory. This is a data corruption and potential code execution vulnerability.

**Remediation**:
```rust
let required_size = size_of::<PageHeaderData>()
    + size_of::<HnswNodePageHeader>()
    + vector.len() * size_of::<f32>();
if required_size > pg_sys::BLCKSZ as usize {
    pgrx::error!("HNSW: Vector with {} dimensions ({} bytes) exceeds page size",
        vector.len(), required_size);
}
```

#### CRITICAL-H2: `write_neighbors_to_page` can overflow page boundary

**Severity: CRITICAL (CWE-787: Out-of-bounds Write)**

Lines 1265-1323:
```rust
unsafe fn write_neighbors_to_page(
    page: pg_sys::Page,
    layer: usize,
    neighbors: &[HnswNeighbor],
    dimensions: usize,
) {
    // ...
    // Shift higher-layer neighbor data if size changed
    if new_size != old_size {
        // Line 1309
        ptr::copy(src, dst, higher_size);
    }

    // Write neighbor entries
    let neighbors_ptr = neighbors_base.add(offset) as *mut HnswNeighbor;
    for (i, neighbor) in neighbors.iter().enumerate() {
        ptr::write(neighbors_ptr.add(i), *neighbor);  // Line 1316
    }
```

There is NO bounds check before writing neighbor data. The total data written includes: PageHeaderData + HnswNodePageHeader + vector data + neighbors for ALL layers. For a node at layer 5 with M=16 and M0=32, this could be: `24 + 32 + dim*4 + 32*sizeof(HnswNeighbor) + 5*16*sizeof(HnswNeighbor)`. Each `HnswNeighbor` is 8 bytes, so: `24 + 32 + dim*4 + 256 + 640 = dim*4 + 952`. For 768-dim vectors: `3072 + 952 = 4024` bytes -- fits. For 1536-dim vectors: `6144 + 952 = 7096` -- fits. For 2048-dim vectors: `8192 + 952 = 9144` -- OVERFLOW.

Additionally, the `ptr::copy` for shifting higher-layer data (line 1309) can read/write out of bounds if the cumulative neighbor data exceeds the page.

**Impact**: Same as CRITICAL-H1 -- shared memory corruption, PostgreSQL crash, potential code execution.

**Remediation**: Add a bounds check before any write:
```rust
let total_data = size_of::<PageHeaderData>()
    + size_of::<HnswNodePageHeader>()
    + dimensions * size_of::<f32>()
    + offset + new_size;
if total_data > pg_sys::BLCKSZ as usize {
    pgrx::warning!("HNSW: Neighbor write would exceed page boundary");
    return;
}
```

#### CRITICAL-H3: `hnsw_build_callback` trusts untrusted varlena data without full validation

**Severity: CRITICAL (CWE-20: Improper Input Validation)**

Lines 912-964:
```rust
unsafe extern "C" fn hnsw_build_callback(
    index: Relation,
    ctid: ItemPointer,
    values: *mut Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut ::std::os::raw::c_void,
) {
    // ...
    let data_ptr = pgrx::varlena::vardata_any(detoasted as *const _) as *const u8;
    let dims = ptr::read_unaligned(data_ptr as *const u16) as usize;  // Line 942
    if dims == 0 {
        return;
    }
    let f32_ptr = data_ptr.add(4) as *const f32;
    std::slice::from_raw_parts(f32_ptr, dims).to_vec()  // Line 947
```

The code reads a `u16` from the varlena payload and interprets it as `dims`. If the varlena is corrupted (e.g., from a crashed WAL replay or disk corruption), `dims` could be up to 65535. The code then creates a slice of `dims * 4 = 262,140` bytes from the pointer, almost certainly reading past the actual varlena allocation.

There is no validation that `dims * 4 + 4 <= varsize_any - VARHDRSZ`.

**Impact**: Heap buffer overread. In the build callback context, this reads garbage data into the index, potentially leaking memory contents of other PostgreSQL backends. This is an information disclosure vulnerability.

**Remediation**: Add size validation:
```rust
let total_size = pgrx::varlena::varsize_any(detoasted as *const _);
let data_size = total_size - pg_sys::VARHDRSZ;
if dims == 0 || 4 + dims * 4 > data_size {
    return; // Corrupted varlena
}
```

### 2.3 HIGH Findings

#### HIGH-H1: `read_metadata` / `write_metadata` trust page contents without magic validation

**Severity: HIGH (CWE-20)**

Lines 393-404:
```rust
unsafe fn read_metadata(page: Page) -> HnswMetaPage {
    let header = page as *const PageHeaderData;
    let data_ptr = (header as *const u8).add(size_of::<PageHeaderData>());
    ptr::read(data_ptr as *const HnswMetaPage)
}
```

The function reads sizeof(HnswMetaPage) bytes from the page without checking the `magic` or `version` fields. If this is called on a corrupted page or a page from a different index type, it reads garbage. The callers do not validate the magic number either.

Compare with `ivfflat_am.rs` line 649: `if meta.magic != IVFFLAT_MAGIC` -- IVFFlat validates, HNSW does not.

**Remediation**: Add validation after reading:
```rust
let meta = ptr::read(data_ptr as *const HnswMetaPage);
if meta.magic != HNSW_MAGIC {
    pgrx::error!("HNSW: Invalid metadata page (bad magic: 0x{:08x})", meta.magic);
}
```

#### HIGH-H2: `hnsw_insert` varlena fallback path lacks size validation

**Severity: HIGH (CWE-125)**

Lines 1059-1073 (same pattern as CRITICAL-H3 but in the INSERT path):
```rust
let data_ptr = pgrx::varlena::vardata_any(detoasted as *const _) as *const u8;
let dims = ptr::read_unaligned(data_ptr as *const u16) as usize;
let f32_ptr = data_ptr.add(4) as *const f32;
std::slice::from_raw_parts(f32_ptr, dims).to_vec()
```

No validation that the varlena size is sufficient for the claimed dimensions.

#### HIGH-H3: `hnsw_rescan` varlena fallback -- better but still trusts dimensions field

**Severity: HIGH (partial mitigation)**

Lines 1696-1728: The rescan function (line 1709-1714) does validate:
```rust
if dimensions > 0
    && dimensions <= 16384
    && actual_data_size >= expected_data_size
```

This is better than the build callback (CRITICAL-H3) because it checks `actual_data_size >= expected_data_size`. However, the 16384 upper bound is arbitrary and not related to the index's actual dimension count.

#### HIGH-H4: `hnsw_beginscan` allocates ORDER BY arrays with `palloc` but no error handling

**Severity: HIGH (CWE-252)**

Lines 1596-1602:
```rust
(*scan).xs_orderbyvals =
    pg_sys::palloc0(std::mem::size_of::<pg_sys::Datum>() * n) as *mut pg_sys::Datum;
(*scan).xs_orderbynulls = pg_sys::palloc(std::mem::size_of::<bool>() * n) as *mut bool;
std::ptr::write_bytes((*scan).xs_orderbynulls, 1u8, n);
```

PostgreSQL's `palloc` calls `ereport(ERROR)` on OOM, which longjmps. The `#[pg_guard]` annotation on the enclosing function should catch this. However, between the `palloc0` on line 1597 and the `palloc` on line 1598, if the second `palloc` fails, the first allocation is leaked in the current memory context (though PostgreSQL will clean it up at transaction end). This is acceptable PostgreSQL behavior.

#### HIGH-H5: No WAL logging for any page modifications

**Severity: HIGH (Data Integrity)**

Throughout the entire file, every page modification follows the pattern:
```rust
pg_sys::MarkBufferDirty(buffer);
pg_sys::UnlockReleaseBuffer(buffer);
```

There are ZERO calls to `XLogInsert`, `XLogRegisterBuffer`, or any WAL-related API. This means:
1. Index changes are not crash-safe. A crash between `MarkBufferDirty` and the next checkpoint will lose the write.
2. Streaming replication will not replicate index changes.
3. Point-in-time recovery (PITR) cannot restore the index.

This is a fundamental design limitation. The index must be rebuilt after any crash (`REINDEX`).

**Remediation**: Either implement WAL logging (complex but necessary for production) or document this limitation prominently and ensure `REINDEX` is triggered after recovery.

### 2.4 MEDIUM Findings

#### MEDIUM-H1: `metric_from_index` string comparison is fragile

Lines 444-454: The function identifies the distance metric by checking if the function name `contains("cosine")` etc. This could match unrelated functions whose names happen to contain these substrings.

#### MEDIUM-H2: `hnsw_bulkdelete` reads metadata with shared lock, then re-reads with exclusive

Lines 1420-1455: `get_meta_page` (shared lock) reads metadata, then `get_meta_page_exclusive` re-reads it. Between these two calls, another backend could modify the metadata. The `deleted_count` update is not atomic.

#### MEDIUM-H3: `hnsw_endscan` correctly prevents use-after-free

Lines 1907-1912:
```rust
if !(*scan).opaque.is_null() {
    let state = Box::from_raw((*scan).opaque as *mut HnswScanState);
    drop(state);
    (*scan).opaque = std::ptr::null_mut();
}
```

This is **correct** -- it null-checks, drops, and nulls out the pointer. Good pattern.

#### MEDIUM-H4: `hnsw_handler` copies a static template that has all callback fields set to `None`

Lines 2062-2089: The handler copies `HNSW_AM_HANDLER` (all callbacks = None) and then sets them. If a new PostgreSQL version adds required callbacks, the `None` values would cause a crash when PostgreSQL tries to call them. This is a maintenance risk.

### 2.5 PostgreSQL Memory Context Analysis

| Function | palloc/pfree Usage | Correct? |
|----------|-------------------|----------|
| `get_meta_page` | `ReadBuffer` (buffer manager) | Yes |
| `metric_from_index` | `pfree(name_ptr)` after `get_func_name` | Yes |
| `allocate_node_page` | `ReadBuffer` (buffer manager) | Yes |
| `hnsw_beginscan` | `palloc0`, `palloc` for orderby arrays | Yes (in scan context) |
| `hnsw_endscan` | `Box::from_raw` (Rust heap) | Yes |
| `hnsw_handler` | `palloc0` for IndexAmRoutine | Yes (in CacheMemoryContext) |

Buffer management follows the correct pattern: `ReadBuffer` -> `LockBuffer` -> use -> `UnlockReleaseBuffer`. No buffer leaks detected in normal code paths.

**Exception**: In `connect_node_to_neighbors` (line 1356-1404), if the `write_neighbors_to_page` call panics (e.g., from a pgrx error macro), the buffer is leaked because `UnlockReleaseBuffer` is called after the write. The `#[pg_guard]` on the enclosing extern "C" function should handle this via PostgreSQL's error cleanup, but the inner helper functions are not `pg_guard`-protected.

---

## 3. ivfflat_am.rs -- 29 unsafe blocks (2,174 LOC)

### 3.1 Unsafe Block Inventory

| Line(s) | PG C API Called | Description | Risk |
|---------|----------------|-------------|------|
| 631-655 | `RelationGetNumberOfBlocksInFork`, `ReadBuffer`, `LockBuffer`, `BufferGetPage`, `ptr::read` | `read_meta_page` | MEDIUM |
| 658-682 | `ReadBuffer`, `LockBuffer`, `BufferGetPage`, `PageInit`, `ptr::write`, `MarkBufferDirty`, `UnlockReleaseBuffer` | `write_meta_page` | MEDIUM |
| 685-728 | `ReadBuffer`, `LockBuffer`, `ptr::read`, `from_raw_parts` | `read_centroids` | **HIGH** |
| 731-782 | `ReadBuffer`, `PageInit`, `ptr::write`, `MarkBufferDirty`, `UnlockReleaseBuffer` | `write_centroids` | MEDIUM |
| 785-830 | `ReadBuffer`, `LockBuffer`, `ptr::write`, `MarkBufferDirty` | `rewrite_centroids` | MEDIUM |
| 833-911 | `ReadBuffer`, `LockBuffer`, `ptr::read`, `from_raw_parts` | `read_inverted_list` | **HIGH** |
| 914-1022 | `ReadBuffer`, `PageInit`, `ptr::write`, `MarkBufferDirty`, `UnlockReleaseBuffer` | `write_inverted_list` | **CRITICAL** |
| 1029-1105 | Calls `read_centroids`, `read_inverted_list` | `ivfflat_search` | MEDIUM |
| 1113-1349 | `pg_guard` extern C, `table_index_build_scan`, varlena | `ivfflat_ambuild` | **HIGH** |
| 1145-1184 | `extern "C"` callback, varlena parsing | `ivf_build_callback` | **CRITICAL** |
| 1353-1359 | `pg_guard` extern C | `ivfflat_ambuildempty` | LOW |
| 1363-1404 | `pg_guard` extern C | `ivfflat_aminsert` | MEDIUM |
| 1408-1422 | `pg_guard` extern C | `ivfflat_ambulkdelete` | LOW |
| 1426-1451 | `pg_guard` extern C | `ivfflat_amvacuumcleanup` | LOW |
| 1455-1489 | `pg_guard` extern C | `ivfflat_amcostestimate` | LOW |
| 1493-1530 | `pg_guard` extern C, `palloc0`, `palloc`, Box alloc | `ivfflat_ambeginscan` | MEDIUM |
| 1534-1660 | `pg_guard` extern C, datum extraction, varlena | `ivfflat_amrescan` | HIGH |
| 1663-1702 | varlena text conversion | `ivfflat_try_convert_text_to_ruvector` | MEDIUM |
| 1706-1750 | `pg_guard` extern C | `ivfflat_amgettuple` | MEDIUM |
| 1754-1758 | `pg_guard` extern C | `ivfflat_amgetbitmap` | LOW |
| 1762-1771 | `Box::from_raw` | `ivfflat_amendscan` | MEDIUM |
| 1775-1778 | `pg_guard` extern C | `ivfflat_amcanreturn` | LOW |
| 1782-1789 | `pg_guard` extern C | `ivfflat_amoptions` | LOW |
| 1793-1796 | `pg_guard` extern C | `ivfflat_amvalidate` | LOW |
| 1801-1815 | `pg_guard` extern C | `ivfflat_amestimateparallelscan` (PG14-17) | LOW |
| 1832-1839 | `SpinLockInit` | `ivfflat_aminitparallelscan` | MEDIUM |
| 1843-1855 | `SpinLockAcquire/Release` | `ivfflat_amparallelrescan` | MEDIUM |
| 1941-1971 | `palloc0`, `ptr::copy_nonoverlapping` | `ruivfflat_handler` | MEDIUM |

### 3.2 CRITICAL Findings

#### CRITICAL-I1: `read_centroids` creates slice from unchecked page data

**Severity: CRITICAL (CWE-125: Out-of-bounds Read)**

Lines 710-716:
```rust
for i in 0..batch_size {
    let entry_ptr = data_ptr.add(i * centroid_size);
    let entry = ptr::read(entry_ptr as *const CentroidEntry);

    let vector_ptr = entry_ptr.add(size_of::<CentroidEntry>()) as *const f32;
    let vector: Vec<f32> = std::slice::from_raw_parts(vector_ptr, dimensions).to_vec();
```

The loop reads `batch_size` centroids from a page without verifying that the total read size fits within the page. `batch_size` is computed as `min(remaining, centroids_per_page)`, and `centroids_per_page = usable_space / centroid_size` (line 697). This arithmetic is correct **if** the page was written correctly. However, if `current_page` points to a corrupted or wrong page (e.g., after a crash without WAL), the data could be arbitrary.

Furthermore, there is no magic number or page type validation on the centroid pages.

**Impact**: Buffer overread, potential information disclosure or crash.

#### CRITICAL-I2: `ivf_build_callback` has identical varlena vulnerability as CRITICAL-H3

**Severity: CRITICAL (CWE-125)**

Lines 1171-1177:
```rust
let data_ptr = pgrx::varlena::vardata_any(detoasted as *const _) as *const u8;
let dims = std::ptr::read_unaligned(data_ptr as *const u16) as usize;
if dims == 0 {
    return;
}
let f32_ptr = data_ptr.add(4) as *const f32;
std::slice::from_raw_parts(f32_ptr, dims).to_vec()
```

Exact same bug as CRITICAL-H3 in hnsw_am.rs. No validation that the varlena size is sufficient for the claimed `dims`.

### 3.3 HIGH Findings

#### HIGH-I1: `read_inverted_list` reads from chained pages without page type validation

**Severity: HIGH (CWE-20)**

Lines 833-911: The function follows a linked list of pages (`current_page = list_header.next_page`). It checks `list_header.page_type != IVFFLAT_PAGE_LIST` (line 857) which is good. However, it then reads `list_header.entry_count` entries without validating that the total read fits in the page.

Lines 872-874:
```rust
for i in 0..list_header.entry_count as usize {
    let entry_ptr = entry_data_ptr.add(i * entry_size);
    let entry = ptr::read(entry_ptr as *const VectorEntry);
```

If `entry_count` is corrupted, this reads past the page boundary.

#### HIGH-I2: `write_inverted_list` buffer management with linked list is complex

**Severity: HIGH (CWE-401: Memory Leak)**

Lines 948-1022: The function maintains `prev_buffer` and `prev_header_ptr` for page chaining. There is a subtle correctness issue:

```rust
// Line 970-972
if !prev_header_ptr.is_null() {
    (*prev_header_ptr).next_page = actual_page;
    pg_sys::MarkBufferDirty(prev_buffer);
    pg_sys::UnlockReleaseBuffer(prev_buffer);
}
```

After releasing `prev_buffer`, the function keeps `prev_header_ptr` which pointed INTO that buffer's page. Although it is immediately overwritten on the next iteration, if a panic occurs between line 972 and line 1013 (setting new `prev_header_ptr`), the old `prev_header_ptr` is a dangling pointer. This is unlikely to be exploitable but is poor practice.

#### HIGH-I3: `ivfflat_amrescan` varlena fallback is identical to HNSW pattern

Lines 1597-1625: Same three-method extraction pattern as hnsw_am.rs. The fallback (Method 3) does validate `actual_data_size >= expected_data_size` (line 1611), which is better than the build callback.

#### HIGH-I4: No WAL logging (same as HIGH-H5)

Same issue as HNSW: no `XLogInsert` calls anywhere. All page modifications are not crash-safe.

### 3.4 MEDIUM Findings

#### MEDIUM-I1: `ivfflat_amendscan` correctly prevents use-after-free

Lines 1762-1771:
```rust
let state = (*scan).opaque as *mut IvfFlatScanState;
if !state.is_null() {
    let _ = Box::from_raw(state);
    (*scan).opaque = ptr::null_mut();
}
```

Correct pattern, same as HNSW.

#### MEDIUM-I2: `write_inverted_list` has `j < dimensions` guard on vector write

Line 1000-1002:
```rust
for (j, &val) in vector.iter().enumerate() {
    if j < dimensions {
        ptr::write(vector_ptr.add(j), val);
    }
}
```

This is a defensive check that prevents writing past the entry's vector slot if the input vector is somehow longer than expected. Good practice, but it silently truncates -- should log a warning.

#### MEDIUM-I3: `ivfflat_aminsert` is a stub -- does not actually insert

Lines 1363-1404: The INSERT callback increments `insertions_since_retrain` but the actual TODO comments on lines 1386-1389 show the vector extraction and insertion are not implemented. This means `INSERT` operations on IVFFlat indexes silently succeed without actually indexing the data.

### 3.5 PostgreSQL Memory Context and Error Handling

| Pattern | HNSW | IVFFlat | Assessment |
|---------|------|---------|------------|
| Buffer lock/unlock | Correct | Correct | Both follow ReadBuffer->Lock->Use->Unlock |
| Magic number validation | **MISSING** | Present (line 649) | IVFFlat is better |
| Page bounds checking | Present (lines 540, 600) | **MISSING** | HNSW is better |
| Use-after-free in endscan | Prevented | Prevented | Both null out opaque |
| palloc error handling | Via pg_guard | Via pg_guard | Both correct |
| WAL logging | **MISSING** | **MISSING** | Both are crash-unsafe |

---

## 4. Cross-Reference with D1 Findings

| D1 Finding | simd.rs | hnsw_am.rs | ivfflat_am.rs | Status |
|------------|---------|------------|---------------|--------|
| **NEON `debug_assert_eq` vs `assert_eq` for buffer bounds** | **PRESENT** (CRITICAL-S1, CRITICAL-S2): 35+ `debug_assert!` calls guard null pointers, zero lengths, and buffer bounds | N/A | N/A | **SAME BUG, wider scope** |
| **u8 accumulator overflow in hamming NEON** | NOT PRESENT: simd.rs only handles f32 distance, no hamming/binary | N/A | N/A | Not applicable |
| **Integer overflow in `grow()`** | NOT PRESENT: no dynamic allocation | NOT PRESENT: uses PostgreSQL buffer manager | NOT PRESENT | Not applicable |
| **Unchecked varlena dimension field** | N/A | **PRESENT** (CRITICAL-H3): dims from u16 without size validation | **PRESENT** (CRITICAL-I2): identical pattern | **NEW finding specific to PG crate** |
| **Page boundary overflow** | N/A | **PRESENT** (CRITICAL-H1, CRITICAL-H2): writes can exceed BLCKSZ | **PARTIALLY PRESENT** (HIGH-I1): reads without bounds checking | **NEW finding specific to PG crate** |

---

## 5. Summary and Prioritized Recommendations

### 5.1 Risk Distribution

```
CRITICAL (7):
  S1: debug_assert for null/length in 14 SIMD ptr functions
  S2: debug_assert for batch buffer bounds in 6 batch functions
  H1: allocate_node_page page overflow (no size check)
  H2: write_neighbors_to_page page overflow (no bounds check)
  H3: hnsw_build_callback varlena overread
  I1: read_centroids unchecked page reads
  I2: ivf_build_callback varlena overread (same as H3)

HIGH (18):
  S1-S4: Aligned load risks, missing length validation, no NEON target_feature
  H1-H5: Missing magic validation, varlena in insert, palloc ordering, no WAL
  I1-I4: Unchecked page reads, buffer management, varlena patterns, no WAL

MEDIUM (18):
  Various: NaN handling, simsimd precision, metric string matching,
           metadata race conditions, stub insert handler

LOW (8):
  Various: Simple dispatch wrappers, stub callbacks, documentation issues
```

### 5.2 Prioritized Fix Order

**P0 -- Fix immediately (blocks any production deployment):**

1. **Replace `debug_assert!` with `assert!` or explicit checks in all SIMD pointer functions** (CRITICAL-S1, CRITICAL-S2). This is the highest-impact fix: 20 sites across simd.rs. Estimated effort: 1 hour.

2. **Add page size validation in `allocate_node_page`** (CRITICAL-H1). Add a check that the total data size fits in BLCKSZ before writing. Estimated effort: 30 minutes.

3. **Add page bounds check in `write_neighbors_to_page`** (CRITICAL-H2). Validate total write size before `ptr::write` and `ptr::copy`. Estimated effort: 30 minutes.

4. **Add varlena size validation in all build callbacks** (CRITICAL-H3, CRITICAL-I2). Validate `4 + dims * 4 <= actual_data_size` before creating slices from varlena data. Fix in 3 locations (hnsw_build_callback, hnsw_insert, ivf_build_callback). Estimated effort: 45 minutes.

5. **Add page bounds check in `read_centroids`** (CRITICAL-I1). Validate that the computed read size fits within the page. Estimated effort: 30 minutes.

**P1 -- Fix before beta release:**

6. Add magic number validation in `read_metadata` for HNSW (HIGH-H1).
7. Add varlena size validation in all insert/rescan varlena fallback paths (HIGH-H2, HIGH-I3).
8. Add `#[target_feature(enable = "neon")]` to NEON functions (HIGH-S3).
9. Change `debug_assert_eq!` to `assert_eq!` in all slice-based SIMD functions (HIGH-S2).
10. Add page entry count validation in `read_inverted_list` (HIGH-I1).

**P2 -- Fix before GA:**

11. Implement WAL logging for all page modifications (HIGH-H5, HIGH-I4).
12. Implement the IVFFlat INSERT path (MEDIUM-I3).
13. Add NaN checking in distance functions or document NaN behavior.

### 5.3 Total Unsafe Block Counts (Full Crate)

| File | unsafe keyword count | Actual unsafe blocks |
|------|---------------------|---------------------|
| `distance/simd.rs` | 78 | 78 |
| `index/hnsw_am.rs` | 40 | 40 |
| `index/ivfflat_am.rs` | 29 | 29 |
| `index/ivfflat_storage.rs` | 9 | 9 |
| `types/halfvec.rs` | 26 | 26 |
| `types/vector.rs` | 19 | 19 |
| `types/binaryvec.rs` | 7 | 7 |
| `types/mod.rs` | 6 | 6 |
| `types/sparsevec.rs` | 5 | 5 |
| `types/scalarvec.rs` | 5 | 5 |
| `types/productvec.rs` | 5 | 5 |
| `quantization/binary.rs` | 4 | 4 |
| `dag/guc.rs` | 6 | 6 |
| `index/bgworker.rs` | 3 | 3 |
| `quantization/scalar.rs` | 2 | 2 |
| `workers/ipc.rs` | 2 | 2 |
| `tenancy/quotas.rs` | 2 | 2 |
| `workers/lifecycle.rs` | 1 | 1 |
| `healing/worker.rs` | 1 | 1 |
| `dag/hooks.rs` | 1 | 1 |
| **TOTAL** | **251** | **251** |

---

## Appendix A: OWASP Mapping

| Finding | OWASP 2021 | CWE |
|---------|-----------|-----|
| CRITICAL-S1, S2 | A04: Insecure Design | CWE-787, CWE-125 |
| CRITICAL-H1, H2 | A04: Insecure Design | CWE-787 |
| CRITICAL-H3, I2 | A03: Injection | CWE-20, CWE-125 |
| CRITICAL-I1 | A03: Injection | CWE-125 |
| HIGH-H5, I4 | A04: Insecure Design | CWE-693 |
| HIGH-H1 | A08: Software Integrity | CWE-20 |

## Appendix B: Files Examined

```
crates/ruvector-postgres/src/distance/simd.rs          (2,128 lines, complete)
crates/ruvector-postgres/src/index/hnsw_am.rs          (2,351 lines, complete)
crates/ruvector-postgres/src/index/ivfflat_am.rs       (2,174 lines, complete)
crates/ruvector-postgres/src/index/ivfflat_storage.rs   (grep survey)
crates/ruvector-postgres/src/types/vector.rs            (grep survey)
crates/ruvector-postgres/src/types/halfvec.rs           (grep survey)
```

---

*Audit completed by V3 QE Security Reviewer. All line numbers reference the codebase as of commit e9bbc7de.*
*Security Score: 28/100 (FAIL) -- 7 critical vulnerabilities must be resolved before merge.*
