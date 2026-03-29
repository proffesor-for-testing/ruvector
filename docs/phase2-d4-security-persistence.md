# Phase 2 Deep Analysis: Domain 4 -- Security & Persistence

**Domain**: D4 Security & Persistence
**Priority**: P1 HIGH
**Crates**: ruvector-postgres, ruvector-server, ruvector-snapshot, ruvector-verified
**Date**: 2026-03-29
**Analyst**: QE Security Scanner (Opus 4.6)

---

## Executive Summary

Domain 4 contains the most security-critical code in the monorepo. The audit found **3 CRITICAL**, **5 HIGH**, **6 MEDIUM**, and **4 LOW** severity findings across the four crates. The most severe issues are: (1) a SQL injection vulnerability in `dag_analyze_plan` that directly interpolates user input into EXPLAIN queries, (2) the complete absence of authentication, authorization, TLS, and rate limiting in ruvector-server, and (3) directory traversal in snapshot ID handling. The SIMD unsafe code is well-structured with proper runtime dispatch, though it universally lacks `// SAFETY:` comments. The PostgreSQL access methods demonstrate solid buffer management with bounds checks that prevent page-boundary overreads.

### Severity Summary

| Severity | Count | Categories |
|----------|-------|------------|
| CRITICAL | 3 | SQL injection, No auth on server, Directory traversal |
| HIGH | 5 | Missing SAFETY comments, SipHash for attestation, CORS wildcard, detoast leaks, debug function SQL injection |
| MEDIUM | 6 | NEON debug_assert-only checks, palloc without pfree tracking, NaN handling in heap ordering, no input size limits on server, no snapshot transfer encryption, tenant admin wildcard policy |
| LOW | 4 | Hardcoded defaults, relaxed atomic ordering, stale in-memory cache, no test coverage on server |

---

## 1. SQL Injection Audit (ruvector-postgres)

### 1.1 CRITICAL: `dag_analyze_plan` -- Direct String Interpolation in SQL

**File**: `crates/ruvector-postgres/src/dag/functions/analysis.rs`, line 24
**Severity**: CRITICAL
**CWE**: CWE-89 (SQL Injection)

```rust
// Line 24 -- user-supplied query_text is directly interpolated
let query = format!("EXPLAIN (FORMAT JSON) {}", query_text);
match client.select(&query, None, None) {
```

The `dag_analyze_plan` function is a `#[pg_extern]` that accepts arbitrary `&str` from SQL callers. The `query_text` parameter is directly interpolated into a string that is then executed via SPI. An attacker with SQL access could execute:

```sql
SELECT dag_analyze_plan('1; DROP TABLE users; --');
```

This would produce: `EXPLAIN (FORMAT JSON) 1; DROP TABLE users; --` which SPI will execute. The same pattern exists in `dag_critical_path` (line 52), `dag_bottlenecks` (line 77), `dag_mincut_analysis` (line 127), `dag_suggest_optimizations` (line 151), `dag_estimate` (line 201), and `dag_learn_from_execution` (line 226) -- all in the same file.

**Remediation**: Use `Spi::run_with_args` with parameterized queries, or at minimum validate that `query_text` is a single SELECT statement and does not contain semicolons or DDL keywords. The EXPLAIN command in PostgreSQL does not support `$1` parameter syntax, so the safest approach is to use `pg_parse_query()` to validate the input is a single SELECT/INSERT/UPDATE/DELETE before prepending EXPLAIN.

### 1.2 HIGH: `ruvector_hnsw_debug` -- SQL Injection via String Escape

**File**: `crates/ruvector-postgres/src/index/hnsw_am.rs`, lines 2123-2153
**Severity**: HIGH
**CWE**: CWE-89

```rust
// Line 2123-2128
let query = format!(
    "SELECT c.oid, c.relname, am.amname \
     FROM pg_class c JOIN pg_am am ON c.relam = am.oid \
     WHERE c.relname = '{}' AND am.amname = 'hnsw'",
    index_name.replace('\'', "''")
);
```

The function uses manual single-quote escaping (`replace('\'', "''")`) which is better than raw interpolation, but is NOT equivalent to proper parameterized queries. PostgreSQL has other escape sequences (e.g., Unicode escapes with `\u`, backslash-escaped strings with `E''`) that could bypass simple quote-doubling depending on `standard_conforming_strings` settings. The same pattern repeats at lines 2148-2153 for the `meta_query`.

**Remediation**: Replace with `Spi::get_one_with_args` using `$1` parameters:
```rust
let index_exists = Spi::get_one_with_args::<bool>(
    "SELECT EXISTS(SELECT 1 FROM pg_class c JOIN pg_am am ON c.relam = am.oid WHERE c.relname = $1 AND am.amname = 'hnsw')",
    vec![(PgBuiltInOids::TEXTOID.oid(), index_name.into_datum())],
);
```

### 1.3 GOOD: Graph/SPARQL Module -- Properly Parameterized

**Files**: `crates/ruvector-postgres/src/graph/mod.rs`, `crates/ruvector-postgres/src/graph/sparql/mod.rs`

All graph and SPARQL persistence queries use `Spi::run_with_args` with `$1`, `$2`, etc. positional parameters. For example:

- `graph/mod.rs` line 155: `Spi::run_with_args("INSERT INTO _ruvector_graphs (name) VALUES ($1) ON CONFLICT DO NOTHING", ...)`
- `graph/mod.rs` line 166-180: `Spi::run_with_args("INSERT INTO _ruvector_nodes ... VALUES ($1, $2, $3, $4) ...", ...)`
- `sparql/mod.rs` line 119: `Spi::run_with_args("INSERT INTO _ruvector_triples ... VALUES ($1, $2, $3, $4, $5, $6) ...", ...)`

All parameterized queries use typed OID parameters (`PgBuiltInOids::TEXTOID`, `INT8OID`, `JSONBOID`) with `.into_datum()` conversion. This is correct and safe.

### 1.4 GOOD: Tenancy Module -- Comprehensive Input Validation

**Files**: `crates/ruvector-postgres/src/tenancy/validation.rs`, `crates/ruvector-postgres/src/tenancy/isolation.rs`

The tenancy module demonstrates exemplary SQL injection prevention:

1. **`validate_identifier()`** (validation.rs:199): Validates table/schema names are 1-63 chars, start with letter/underscore, contain only `[a-zA-Z0-9_]`, and are not reserved words.
2. **`validate_tenant_id()`** (validation.rs:153): Similar validation for tenant IDs with additional hyphen allowance.
3. **`quote_identifier()`** (validation.rs:282): Wraps identifiers in double quotes with escaped internal double quotes.
4. **`escape_string_literal()`** (validation.rs:264): Properly doubles single quotes.
5. **`safe_partition_name()`** (validation.rs:289): Combines validation + sanitization for partition names.

All SQL generation in `isolation.rs` (e.g., RLS policy creation at line 131, partition creation) uses validated/quoted identifiers. This module should be the template for the DAG module.

---

## 2. Unsafe Audit: Postgres SIMD (`distance/simd.rs`, 78 blocks)

### 2.1 Architecture Overview

The file is 2,129 lines implementing distance functions across four SIMD tiers:
- **AVX-512** (16 floats/iteration, behind `simd-avx512` feature flag)
- **AVX2** (8 floats/iteration, with 4x unrolled variants processing 32 floats/iteration)
- **ARM NEON** (4 floats/iteration)
- **simsimd** integration (auto-dispatched, used for common dimensions 384/768/1536/3072)
- **Scalar fallback** (pointer-based and slice-based)

Runtime dispatch is performed via `is_x86_feature_detected!()` and `is_neon_available()`.

### 2.2 HIGH: No `// SAFETY:` Comments Anywhere

**Severity**: HIGH
**CWE**: CWE-676 (Use of Potentially Dangerous Function)

All 78 unsafe blocks in `simd.rs` lack `// SAFETY:` comments, which is a violation of Rust best practices (Clippy lint `clippy::undocumented_unsafe_blocks`). The `# Safety` doc comments on public functions document the requirements, but the actual `unsafe` call sites within function bodies have no local safety justification.

Examples requiring documentation:
- Line 138: `_mm512_loadu_ps(a.add(offset))` -- why is `a.add(offset)` valid?
- Line 463: `_mm256_load_ps(a.add(offset))` -- alignment-required load, guarded by `is_avx2_aligned()` check (correct, but undocumented)
- Line 1514: `vld1q_f32(a.as_ptr().add(offset))` -- NEON load, why is offset in bounds?

### 2.3 Bounds Checking Analysis (Top 20 Unsafe Blocks)

| Line | Function | Guard | Status |
|------|----------|-------|--------|
| 130 | `l2_distance_ptr_avx512` | `debug_assert!(!a.is_null() && !b.is_null() && len > 0)` | **debug_assert only** |
| 164 | `cosine_distance_ptr_avx512` | Same | **debug_assert only** |
| 213 | `inner_product_ptr_avx512` | Same | **debug_assert only** |
| 244 | `manhattan_distance_ptr_avx512` | Same | **debug_assert only** |
| 453 | `l2_distance_ptr_avx2` | Same | **debug_assert only** |
| 504 | `cosine_distance_ptr_avx2` | Same | **debug_assert only** |
| 569 | `inner_product_ptr_avx2` | Same | **debug_assert only** |
| 610 | `manhattan_distance_ptr_avx2` | Same | **debug_assert only** |
| 658 | `l2_distance_ptr_scalar` | Same | **debug_assert only** |
| 675 | `cosine_distance_ptr_scalar` | Same | **debug_assert only** |
| 704 | `inner_product_ptr_scalar` | Same | **debug_assert only** |
| 720 | `manhattan_distance_ptr_scalar` | Same | **debug_assert only** |
| 851-928 | Batch functions | `debug_assert!(results.len() >= vectors.len())` | **debug_assert only** |
| 1505 | `euclidean_distance_neon` | No explicit assert on `a.len() == b.len()` | **Missing** |
| 1532 | `cosine_distance_neon` | No explicit assert | **Missing** |
| 1571 | `inner_product_neon` | No explicit assert | **Missing** |
| 1597 | `manhattan_distance_neon` | No explicit assert | **Missing** |
| 1625 | `manhattan_distance_ptr_neon` | `debug_assert!(!a.is_null() && !b.is_null() && len > 0)` | **debug_assert only** |

**Key Finding**: ALL pointer-based functions use `debug_assert!` only, which is stripped in release builds. If a caller passes `len=0` or null pointers in release mode, the result is undefined behavior (reading from null or computing `chunks = 0 / N` which is fine but the pattern is fragile). The slice-based NEON functions (`euclidean_distance_neon` etc.) have NO length-equality check at all -- they use `a.len()` for loop bounds and `a[i]`/`b[i]` for remainder, which would panic in debug mode and UB in release mode if `b.len() < a.len()`.

### 2.4 MEDIUM: NEON Functions Missing Length Equality Assertion

**File**: `crates/ruvector-postgres/src/distance/simd.rs`, lines 1505, 1532, 1571, 1597
**Severity**: MEDIUM (same issue as D1's NEON findings)

The NEON slice-based functions do not assert `a.len() == b.len()`. While the callers (wrappers) may guarantee this, the functions themselves are `unsafe` and should validate their preconditions. Compare with the optimized dispatch functions (`l2_distance_optimized` at line 1183) which correctly call `debug_assert_eq!(a.len(), b.len())`.

### 2.5 Alignment Check: Correct

The AVX2 pointer functions at lines 458-476 correctly check alignment before using aligned loads:
```rust
let use_aligned = is_avx2_aligned(a, b);
if use_aligned {
    let va = _mm256_load_ps(a.add(offset));  // Aligned load (requires 32-byte alignment)
} else {
    let va = _mm256_loadu_ps(a.add(offset)); // Unaligned load (always safe)
}
```
This is correct. `_mm256_load_ps` requires 32-byte alignment and would cause SIGBUS/SIGSEGV on misaligned addresses. The fallback to `_mm256_loadu_ps` handles the unaligned case.

### 2.6 Comparison with D1 SIMD

The same `debug_assert!`-only pattern and missing NEON length checks found in D1's `ruvector-cnn` crate are present here in D4. This is a systemic pattern across the codebase. The risk is identical: in release mode, passing mismatched-length vectors to NEON functions could cause out-of-bounds reads.

---

## 3. Access Method Audit (`hnsw_am.rs` at 2,351 LOC, `ivfflat_am.rs` at 2,174 LOC)

### 3.1 PostgreSQL C API Calls

The HNSW access method makes extensive use of the PostgreSQL buffer manager:
- `pg_sys::ReadBuffer` (lines 357, 365, 376-379, 467, 511, 532, 577)
- `pg_sys::LockBuffer` (lines 358, 366, 383-385, 470, 512, 533, 578)
- `pg_sys::BufferGetPage` (lines 359, 367, 388, 471, 513, 534, 579)
- `pg_sys::UnlockReleaseBuffer` (lines 497, 519, 562, 625, and many more)
- `pg_sys::MarkBufferDirty` (lines 496, 878, 889, 1017, 1092)
- `pg_sys::PageInit` (lines 474, 846, 1012)
- `pg_sys::table_index_build_scan` (line 986)

### 3.2 Buffer Lock/Unlock Balance

Every `ReadBuffer` + `LockBuffer` pair in the HNSW code has a corresponding `UnlockReleaseBuffer`:

| Function | ReadBuffer Line | UnlockReleaseBuffer Line | Balanced? |
|----------|----------------|--------------------------|-----------|
| `get_meta_page` | 357 | Caller's responsibility | Yes (callers check) |
| `get_meta_page_exclusive` | 365 | Caller's responsibility | Yes |
| `allocate_node_page` | 467 | 497 | **Yes** |
| `read_node_header` | 511 | Caller via buffer return | **Partial** -- buffer returned to caller |
| `read_vector` | 532 | 552 (early return), 562 (normal) | **Yes** |
| `read_neighbors` | 577 | 616 (early return), 625 (normal) | **Yes** |
| `hnsw_search` | via read_node_header | 726 (immediate release) | **Yes** |

**Finding**: The `read_node_header` function returns a `(HnswNodePageHeader, Buffer)` tuple, requiring the caller to release the buffer. All callers (e.g., `hnsw_search` at line 726, `mark_node_deleted` at line 1471) do call `UnlockReleaseBuffer`. The pattern is correct but fragile -- if a future caller forgets the release, it will cause a buffer leak (not a crash, but degraded performance).

### 3.3 GOOD: Page Boundary Bounds Checks

The code at lines 539-553 and 600-617 includes explicit bounds checks that prevent reading past page boundaries:

```rust
// Line 539-553 (read_vector)
let total_read_end = size_of::<PageHeaderData>()
    + size_of::<HnswNodePageHeader>()
    + dimensions * size_of::<f32>();
if total_read_end > page_size {
    pgrx::warning!("HNSW: Vector read would exceed page boundary...");
    pg_sys::UnlockReleaseBuffer(buffer);
    return None;
}
```

This was added to fix issue #164 (segfault). The same check exists in `read_neighbors` at line 600. This is correct and necessary.

### 3.4 MEDIUM: No SPI Error Handling in `hnsw_build`

The `hnsw_build` function does not use SPI at all (correct for an AM build callback), but the `hnsw_insert` callback at line 1023 holds the metadata page exclusively locked during the entire insert operation (lines 1041-1093). If the insert involves many neighbor connections at multiple layers, this could block other concurrent inserts. This is a correctness concern rather than a security issue, but could lead to lock contention under high write load.

### 3.5 MEDIUM: `pg_detoast_datum` Memory Leak

**File**: `crates/ruvector-postgres/src/index/hnsw_am.rs`, lines 937, 1064, 1698, 1777
**Severity**: MEDIUM
**CWE**: CWE-401 (Memory Leak)

The `pg_detoast_datum` calls create detoasted copies that are never explicitly freed with `pfree()`. While PostgreSQL's memory context system will reclaim this memory at the end of the transaction, repeated inserts within a single transaction (e.g., bulk COPY) will accumulate detoasted copies until transaction end. Example:

```rust
// Line 1064
let detoasted = pg_sys::pg_detoast_datum(raw_ptr);
// ... used but never pfree'd
```

Only `metric_from_index` at line 456 correctly calls `pg_sys::pfree(name_ptr as *mut _)`.

### 3.6 Transaction/Snapshot Management

The `build_index_from_heap` function at line 968 passes `std::ptr::null_mut()` as the snapshot parameter to `table_index_build_scan`, which means it uses an MVCC snapshot. This is correct for index builds. The search functions do not manage transactions/snapshots directly -- they rely on the ambient transaction context, which is correct for AM scan callbacks.

### 3.7 ivfflat_am.rs

The IVFFlat access method follows the same patterns as HNSW. It does NOT use SPI at all (no SQL injection risk). It has the same buffer management patterns with similar correctness. It uses k-means clustering for centroid computation, which is all done in Rust without FFI to C libraries.

---

## 4. Authentication in ruvector-server

### 4.1 CRITICAL: No Authentication Whatsoever

**File**: `crates/ruvector-server/src/lib.rs`
**Severity**: CRITICAL
**CWE**: CWE-306 (Missing Authentication for Critical Function)

The server has **zero authentication mechanisms**:
- No JWT support
- No API key support
- No basic auth
- No mTLS
- No middleware extractors for auth

All endpoints (create/delete collections, upsert/search/get points) are publicly accessible to anyone who can reach the server port.

The README explicitly acknowledges this:
```
- **Rate Limiting**: Request rate limiting (planned)
- **Authentication**: API key auth (planned)
```

### 4.2 CRITICAL: CORS Allows All Origins, Methods, and Headers

**File**: `crates/ruvector-server/src/lib.rs`, lines 85-89
**Severity**: HIGH (CRITICAL if server is exposed to the internet)
**CWE**: CWE-942 (Permissive Cross-domain Policy)

```rust
let cors = CorsLayer::new()
    .allow_origin(Any)
    .allow_methods(Any)
    .allow_headers(Any);
```

`allow_origin(Any)` allows any website to make requests to the API. Combined with no authentication, any malicious webpage could read/write/delete vector collections via cross-origin requests from a user's browser.

### 4.3 No TLS/HTTPS

The server binds a plain TCP listener with no TLS support:
```rust
let listener = tokio::net::TcpListener::bind(addr).await?;
```

There is no `axum_server::tls_rustls` or similar TLS configuration. All data (including vector embeddings and metadata) is transmitted in plaintext.

### 4.4 No Rate Limiting

No rate limiting middleware exists. An attacker can flood the server with requests, potentially causing OOM via large vector insertions or excessive collection creation.

### 4.5 MEDIUM: No Input Validation on Request Bodies

**File**: `crates/ruvector-server/src/routes/points.rs`

The `SearchRequest` accepts a `vector: Vec<f32>` with no size limit. An attacker could send:
```json
{"vector": [1.0, 2.0, ...], "k": 999999999}
```

The `k` parameter defaults to 10 but has no upper bound. The `vector` field has no dimension validation against the collection's expected dimensions (though `ruvector-core` may validate this internally).

### 4.6 No Tests

The ruvector-server crate has **zero test functions**. No `#[test]`, `#[tokio::test]`, or `#[cfg(test)]` blocks exist anywhere in the crate.

---

## 5. Snapshot Integrity (ruvector-snapshot)

### 5.1 GOOD: SHA-256 Checksum Verification

**File**: `crates/ruvector-snapshot/src/storage.rs`, lines 72-76, 152-158

Snapshots use SHA-256 checksums computed before compression and verified on load:
```rust
// Save: checksum of serialized data before compression
let checksum = Self::calculate_checksum(&serialized);

// Load: verify after decompression
let actual_checksum = Self::calculate_checksum(&decompressed);
if actual_checksum != snapshot.checksum {
    return Err(SnapshotError::InvalidChecksum { expected, actual });
}
```

The checksum is stored in a separate metadata JSON file and verified before deserialization. A corrupt snapshot is detected and rejected before any data is returned.

### 5.2 CRITICAL: Directory Traversal in Snapshot ID

**File**: `crates/ruvector-snapshot/src/storage.rs`, lines 42-47
**Severity**: CRITICAL
**CWE**: CWE-22 (Path Traversal)

```rust
fn snapshot_path(&self, id: &str) -> PathBuf {
    self.base_path.join(format!("{}.snapshot.gz", id))
}
fn metadata_path(&self, id: &str) -> PathBuf {
    self.base_path.join(format!("{}.metadata.json", id))
}
```

The snapshot ID is used directly in path construction without sanitization. An attacker with API access could pass:
```
id = "../../etc/passwd"  // reads /etc/passwd.metadata.json
id = "../../../tmp/evil" // writes outside base_path
```

The `load` function at line 132 reads the metadata file and then the snapshot file at the constructed path. The `delete` function at line 198 calls `fs::remove_file` at the constructed path. Both are exploitable.

**Remediation**: Validate that the snapshot ID contains only alphanumeric characters, hyphens, and underscores (similar to the UUID format it's generated with). Verify the canonicalized path starts with `base_path`.

### 5.3 GOOD: Corrupt Snapshot Cannot Crash System

A corrupt compressed file fails at decompression (gzip returns error), and a corrupt serialized file fails at bincode deserialization. Both return `SnapshotError` rather than panicking. The checksum verification provides an additional layer of defense.

### 5.4 MEDIUM: No Snapshot Transfer Encryption

Snapshot files are stored as gzip-compressed bincode. There is no encryption-at-rest. If the storage backend is a shared filesystem, other processes/users could read the snapshot data. The `SnapshotStorage` trait supports async/pluggable backends, but the only implementation (`LocalStorage`) has no encryption support.

### 5.5 GOOD: Dimension Validation on Create

**File**: `crates/ruvector-snapshot/src/manager.rs`, lines 32-42

The `create_snapshot` method validates that all vectors have the expected dimension:
```rust
for (idx, vector) in snapshot_data.vectors.iter().enumerate() {
    if vector.vector.len() != expected_dim {
        return Err(SnapshotError::storage(format!(
            "Vector {} has dimension {} but expected {}",
            idx, vector.vector.len(), expected_dim
        )));
    }
}
```

---

## 6. Crypto Audit (ruvector-verified)

### 6.1 Architecture

The `ruvector-verified` crate provides formal verification primitives, not traditional cryptography. It implements:
- Proof environments with symbol tables and term allocation
- Proof attestations with hash-based integrity
- Bump-allocating term arenas with deduplication caches
- No actual encryption, signing, or key management

### 6.2 HIGH: SipHash-2-4 Used for Security-Relevant Hashing

**File**: `crates/ruvector-verified/src/proof_store.rs`, lines 108-121
**Severity**: HIGH (if attestations are used for trust decisions)
**CWE**: CWE-328 (Reversible One-Way Hash)

```rust
fn siphash_256(data: &[u8]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for (i, chunk) in result.chunks_exact_mut(8).enumerate() {
        let mut hasher = DefaultHasher::new(); // SipHash-2-4
        (i as u64).hash(&mut hasher);
        data.hash(&mut hasher);
        chunk.copy_from_slice(&hasher.finish().to_le_bytes());
    }
    result
}
```

`DefaultHasher` is `SipHash-2-4`, a keyed MAC designed for HashDoS resistance, NOT for cryptographic integrity. The "key" is Rust's built-in seed (which changes between process invocations via `RUST_HASH_SEED`). This means:
1. Attestation hashes are not reproducible across process restarts
2. SipHash-2-4 is not collision-resistant against a determined attacker
3. The domain separation (`(i as u64).hash(&mut hasher)`) is minimal

For proof attestations that are meant to provide tamper detection, a cryptographic hash (SHA-256 via `sha2` crate, already used in ruvector-snapshot) should be used instead.

### 6.3 GOOD: No Custom Crypto

The crate does not implement any custom encryption, signing, or key derivation. The `sha2` dependency is used in `ruvector-snapshot` for checksums. No timing side channels exist because there are no secret-key comparisons.

### 6.4 GOOD: No Key Management Concerns

There are no cryptographic keys to manage. The `ProofEnvironment` uses in-memory term counters and caches that are not persisted across restarts.

### 6.5 GOOD: No Unsafe Code

The `ruvector-verified` crate (excluding feature-gated `fast_arena`) contains zero `unsafe` blocks. The `fast_arena.rs` uses `RefCell` for interior mutability rather than raw pointers.

---

## 7. Test Analysis

### 7.1 Test Coverage by Crate

| Crate | Test Count | Test Types | Coverage Assessment |
|-------|-----------|------------|---------------------|
| ruvector-postgres/distance/simd.rs | 17 | Unit | GOOD: tests all distance metrics, remainder handling, feature detection |
| ruvector-postgres/index/hnsw_am.rs | 8 | Unit + pg_test | PARTIAL: struct sizes, defaults, ordering; no integration tests for insert/search |
| ruvector-postgres/index/ivfflat_am.rs | ~9 | Unit + pg_test | PARTIAL: similar to HNSW |
| ruvector-postgres/graph/mod.rs | 1 | Unit | LOW: only tests in-memory registry |
| ruvector-postgres/tenancy/validation.rs | ~10 | Unit | GOOD: tests valid/invalid IDs, injection attempts |
| ruvector-server | 0 | None | **ZERO** tests |
| ruvector-snapshot | 14 | Unit + Tokio | GOOD: roundtrip, compression, checksum, lifecycle |
| ruvector-verified | 83 | Unit | EXCELLENT: comprehensive coverage of all modules |

### 7.2 Critical Test Gaps

1. **ruvector-server**: Zero test coverage. No integration tests for CORS behavior, error handling, or route functionality.

2. **SQL Injection**: The `dag/functions/analysis.rs` functions have `#[pg_test]` tests but they only test with benign inputs like `"SELECT 1"`. No tests attempt injection payloads.

3. **SIMD NEON**: NEON tests are only compiled on `aarch64`. On x86_64 CI, NEON code paths are completely untested. The NEON wrappers at lines 1714-1749 have no conditional test compilation.

4. **HNSW Search Integration**: No tests exercise the full search path (insert vectors, then search). The `#[pg_test]` tests would require a running PostgreSQL instance.

5. **Snapshot Path Traversal**: No tests attempt to create/load snapshots with malicious IDs containing `../`.

6. **Buffer Leak Detection**: No tests verify that all `ReadBuffer` calls are matched with `UnlockReleaseBuffer`.

---

## 8. Additional Findings

### 8.1 MEDIUM: NaN Handling in Heap Ordering

**File**: `crates/ruvector-postgres/src/index/hnsw_am.rs`, lines 304-318, 340-348

The `SearchCandidate` and `ResultCandidate` ordering implementations use:
```rust
other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
```

If `distance` is NaN (e.g., from cosine distance of zero-norm vectors), `partial_cmp` returns `None`, and `unwrap_or(Ordering::Equal)` treats NaN as equal to everything. This can corrupt the binary heap ordering, causing incorrect search results. The cosine distance functions do handle zero denominators (returning 1.0), but NaN could still arise from inf * 0 or similar edge cases.

### 8.2 MEDIUM: Tenant Admin Wildcard Policy

**File**: `crates/ruvector-postgres/src/tenancy/isolation.rs`, lines 152-155

```sql
CREATE POLICY ruvector_admin_wildcard ON {table}
    FOR SELECT
    USING (current_setting('ruvector.tenant_id', true) = '*');
```

Any session that sets `ruvector.tenant_id = '*'` bypasses tenant isolation. This is documented as intentional for admin queries, but it means any user with `SET` privileges can bypass RLS. The `true` parameter to `current_setting` means it returns NULL (not error) if not set, which is correct, but the wildcard bypass should be restricted to a specific role.

### 8.3 LOW: Relaxed Atomic Ordering for Statistics

**File**: `crates/ruvector-postgres/src/index/hnsw_am.rs`, lines 81-83, 2110-2112

Statistics counters use `AtomicOrdering::Relaxed`:
```rust
TOTAL_SEARCHES.fetch_add(1, AtomicOrdering::Relaxed);
```

This is acceptable for statistics (approximate counts are fine), but the `ruhnsw_reset_stats` function also uses `Relaxed` for the store. In theory, another thread could see a partially-reset state. For statistics this is harmless.

### 8.4 LOW: Stale In-Memory Graph Cache

**File**: `crates/ruvector-postgres/src/graph/mod.rs`, line 22

The `GRAPH_REGISTRY` is a `Lazy<DashMap>` that caches graph structures in memory. If another PostgreSQL backend modifies the graph tables directly (via SQL), the cache becomes stale. There is no cache invalidation mechanism. This could lead to data inconsistency between backends.

---

## 9. Remediation Priority

### Immediate (P0 -- Fix Before Release)

1. **SQL Injection in `dag_analyze_plan`** (analysis.rs:24): Replace `format!("EXPLAIN ...")` with input validation. Reject semicolons and DDL keywords, or use `pg_parse_query()` to validate single-statement input.

2. **Directory Traversal in Snapshot ID** (storage.rs:42-47): Validate snapshot IDs against `^[a-zA-Z0-9_-]+$` regex. Verify canonicalized path stays within `base_path`.

3. **Authentication on ruvector-server**: Implement at minimum API key authentication via an `Authorization` header extractor middleware.

### Short-Term (P1 -- Next Sprint)

4. **SQL Injection in `ruvector_hnsw_debug`** (hnsw_am.rs:2123): Switch to parameterized queries.

5. **CORS Restrictions**: Replace `allow_origin(Any)` with configurable allowed origins.

6. **`// SAFETY:` Comments**: Add safety justifications to all 78 unsafe blocks in simd.rs and all 40 in hnsw_am.rs.

7. **SipHash -> SHA-256 for Attestations**: Replace `DefaultHasher` in proof_store.rs with `sha2::Sha256`.

8. **TLS Support**: Add `axum_server::tls_rustls` support with configurable certificate paths.

### Medium-Term (P2 -- Next Quarter)

9. **Rate Limiting**: Add `tower-governor` or similar rate limiting middleware to ruvector-server.

10. **NEON Length Assertions**: Add `debug_assert_eq!(a.len(), b.len())` to all NEON slice-based functions.

11. **Input Size Limits**: Add maximum vector dimension and k limits to server request handlers.

12. **`pfree` for Detoasted Data**: Add explicit `pfree` calls after `pg_detoast_datum` usage.

13. **Server Tests**: Add integration tests covering all routes, error cases, and CORS behavior.

14. **Tenant Wildcard Policy**: Restrict the `*` wildcard policy to a specific PostgreSQL role.

---

## 10. Cross-Domain Observations

1. **SIMD Pattern Consistency**: The `debug_assert!`-only pattern in D4 matches D1 exactly. This should be addressed globally via a codebase-wide policy.

2. **SQL Safety Asymmetry**: The tenancy module is exemplary while the DAG module is critically vulnerable. Both are in the same crate. This suggests the code was written by different contributors without a shared security review process.

3. **Server Maturity**: ruvector-server is clearly early-stage (no auth, no TLS, no tests, no rate limiting). It should not be deployed in any environment accessible beyond localhost without significant hardening.

4. **Snapshot module is well-designed**: Good use of SHA-256, gzip compression, bincode serialization, and proper error handling. Only the path traversal issue needs fixing.

5. **ruvector-verified is the highest-quality crate in D4**: 83 tests, zero unsafe, clean error handling. The SipHash issue is the only concern and is more of a design question than a bug.

---

## Appendix: Files Examined

| File | Lines | Unsafe Blocks | SQL Queries | Tests |
|------|-------|---------------|-------------|-------|
| `ruvector-postgres/src/distance/simd.rs` | 2,129 | 78 | 0 | 17 |
| `ruvector-postgres/src/index/hnsw_am.rs` | 2,351 | 40 | 3 (2 vulnerable) | 8 |
| `ruvector-postgres/src/index/ivfflat_am.rs` | 2,174 | ~29 | 0 | ~9 |
| `ruvector-postgres/src/dag/functions/analysis.rs` | 267 | 0 | 1 (vulnerable) | 2 |
| `ruvector-postgres/src/graph/mod.rs` | 300 | 0 | 12 (all safe) | 1 |
| `ruvector-postgres/src/graph/sparql/mod.rs` | 200 | 0 | 8 (all safe) | 0 |
| `ruvector-postgres/src/tenancy/validation.rs` | 400 | 0 | 0 | 10 |
| `ruvector-postgres/src/tenancy/isolation.rs` | 650 | 0 | 4 (all safe) | 0 |
| `ruvector-server/src/lib.rs` | 126 | 0 | 0 | 0 |
| `ruvector-server/src/routes/collections.rs` | 122 | 0 | 0 | 0 |
| `ruvector-server/src/routes/points.rs` | 123 | 0 | 0 | 0 |
| `ruvector-server/src/routes/health.rs` | 47 | 0 | 0 | 0 |
| `ruvector-server/src/state.rs` | 61 | 0 | 0 | 0 |
| `ruvector-server/src/error.rs` | 77 | 0 | 0 | 0 |
| `ruvector-snapshot/src/snapshot.rs` | 196 | 0 | 0 | 3 |
| `ruvector-snapshot/src/storage.rs` | 277 | 0 | 0 | 4 |
| `ruvector-snapshot/src/manager.rs` | 294 | 0 | 0 | 5 |
| `ruvector-snapshot/src/error.rs` | 53 | 0 | 0 | 0 |
| `ruvector-verified/src/lib.rs` | 232 | 0 | 0 | 7 |
| `ruvector-verified/src/proof_store.rs` | 266 | 0 | 0 | 10 |
| `ruvector-verified/src/fast_arena.rs` | 291 | 0 | 0 | 11 |
