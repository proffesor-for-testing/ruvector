# Phase 3: Consolidated Security Posture Assessment

**Date**: 2026-03-29
**Auditor**: QE Security Auditor v3 (Opus 4.6)
**Scope**: All 10 domains of the RuVector monorepo
**Basis**: Phase 1 automated scans + Phase 2 deep domain audits
**Classification**: OWASP Top 10 2021 + CWE aligned

---

## Executive Summary

The RuVector monorepo contains **37 confirmed security findings** ranked by severity. The attack surface spans 7 external-facing interfaces: HTTP server, LLM inference server, PostgreSQL extension, WASM bindings, Node.js bindings, CLI tools, and SvelteKit UI. The most critical risks are concentrated in Domain 4 (Security & Persistence), where SQL injection, directory traversal, missing authentication, and page boundary overflows create a chain of exploitable vulnerabilities in the PostgreSQL extension and vector server.

A systemic `debug_assert` pattern affects approximately 95 sites across D1, D4, D5, and D6, stripping all safety checks from release builds. This single class of bug accounts for the majority of memory safety findings.

The supply chain has 6 known CVEs (2 HIGH, 1 MEDIUM, 3 unscored) and 2 unsoundness warnings in allowed dependencies.

**Overall Risk Rating: HIGH** -- Multiple CRITICAL findings exist that must be resolved before any production deployment.

| Severity | Count | Domains Affected |
|----------|------:|-----------------|
| CRITICAL | 8 | D4, D9 |
| HIGH | 14 | D1, D4, D5, D6, D8, D9 |
| MEDIUM | 10 | D4, D5, D8, D9 |
| LOW | 5 | D4, D8, D9 |

---

## 1. Master Vulnerability List

### CRITICAL Findings

| ID | Domain(s) | CWE | Description | Exploitability | Impact | Status | Recommended Fix |
|----|-----------|-----|-------------|---------------|--------|--------|----------------|
| SEC-001 | D4 | CWE-89 | **SQL injection in `dag_analyze_plan`**: User-supplied `query_text` is directly interpolated into `format!("EXPLAIN (FORMAT JSON) {}", query_text)` and executed via SPI. 7 functions in `analysis.rs` share this pattern. | HIGH -- Any user with SQL access can inject arbitrary statements via `SELECT dag_analyze_plan('1; DROP TABLE ...')`. | CRITICAL -- Full database compromise: DDL execution, data exfiltration, privilege escalation within PostgreSQL. | Confirmed | Parse input with `pg_parse_query()` to validate it is a single DML statement before prepending EXPLAIN. Parameterized queries cannot be used for EXPLAIN. |
| SEC-002 | D4 | CWE-306 | **No authentication on ruvector-server**: Zero auth mechanisms (no JWT, API key, basic auth, mTLS). All endpoints publicly accessible. | TRIVIAL -- Any network-reachable client can read/write/delete all collections and vectors. | CRITICAL -- Complete data loss or exfiltration. Attacker can overwrite embeddings to poison downstream ML pipelines. | Confirmed | Implement API key authentication middleware as first priority. Add mTLS for production deployments. |
| SEC-003 | D4 | CWE-22 | **Directory traversal in snapshot IDs**: `snapshot_path()` joins unsanitized `id` parameter directly into `PathBuf`. IDs like `../../etc/passwd` escape the base directory. Affects `load`, `save`, and `delete` operations. | HIGH -- Requires API access (trivial if SEC-002 is unfixed). | HIGH -- Arbitrary file read/write/delete on the server filesystem. | Confirmed | Validate snapshot IDs against `[a-zA-Z0-9_-]+` regex. Canonicalize resolved path and verify it starts with `base_path`. |
| SEC-004 | D4 | CWE-787 | **Page boundary overflow in `allocate_node_page`**: No check that vector data + header fits within PostgreSQL's 8KB page (`BLCKSZ`). Vectors with >2,034 dimensions (e.g., GPT-4's 3,072-dim embeddings) write past page boundary. | MEDIUM -- Triggered by inserting high-dimensional vectors (GPT-4, text-embedding-3-large). Requires index creation access. | CRITICAL -- Corrupts PostgreSQL shared memory. Can crash postmaster or enable arbitrary code execution within the database server process. | Confirmed | Add size validation: `assert!(header_size + vector.len() * 4 <= BLCKSZ)` before any page write. |
| SEC-005 | D4 | CWE-787 | **Page boundary overflow in `write_neighbors_to_page`**: No bounds check before writing neighbor data. High-layer nodes with M=16 and >2,048-dim vectors overflow the page. | MEDIUM -- Same trigger as SEC-004, plus requires high-connectivity HNSW graphs. | CRITICAL -- Same as SEC-004: shared memory corruption in PostgreSQL. | Confirmed | Add cumulative size check before writing: verify `header + vector + all_layer_neighbors <= BLCKSZ`. |
| SEC-006 | D4 | CWE-125 | **Varlena overread in `hnsw_build_callback`**: Reads `dims` from varlena payload as `u16` without validating that `dims * 4 + 4 <= varsize - VARHDRSZ`. Corrupted WAL replay or disk data can claim up to 65,535 dimensions, reading 256KB past the actual allocation. | LOW -- Requires corrupted on-disk data or malicious WAL. Cannot be triggered by normal SQL operations. | HIGH -- Heap buffer overread in PostgreSQL backend. Information disclosure (leaks memory contents of other backends). Index corruption from garbage data. | Confirmed | Validate: `if 4 + dims * 4 > data_size { return; }` before constructing the slice. |
| SEC-007 | D4 | CWE-787 | **`debug_assert` systemic bug in D4 `simd.rs`**: All 14 pointer-based SIMD functions and 6 batch functions guard preconditions with `debug_assert!`, stripped in release builds. Batch functions write to `results[i]` without runtime bounds check -- heap buffer overflow if `results.len() < vectors.len()`. | MEDIUM -- Requires a caller to pass mismatched buffer sizes. Internal to the PostgreSQL extension, so direct external exploitation is unlikely, but a bug in index code could trigger it. | CRITICAL -- Heap corruption inside PostgreSQL backend process. The batch function variant (CWE-787 write) is more severe than the read variant. | Confirmed | Replace all `debug_assert!` with `assert!` in SIMD functions. For batch functions, add: `assert!(results.len() >= vectors.len())`. |
| SEC-008 | D9 | CWE-79 | **XSS via `{@html token.html}` in MarkdownBlock.svelte**: Renders Marked tokenizer output directly into the DOM without DOMPurify sanitization. CodeBlock.svelte correctly uses DOMPurify for the same pattern. A bypass of the Marked tokenizer's sanitization rules (HTML entity edge cases, parser bugs) would achieve DOM-level XSS. | MEDIUM -- Requires finding a bypass in the Marked tokenizer's href/tag sanitization. The tokenizer does block `javascript:` and `data:text/html` URIs, but does not use a formal allowlist. | HIGH -- Full XSS in the chat UI: session hijacking, credential theft, conversation exfiltration. The UI handles OIDC tokens and user conversations. | Confirmed | Wrap output in `DOMPurify.sanitize(token.html)` as defense-in-depth, matching the CodeBlock pattern. |

### HIGH Findings

| ID | Domain(s) | CWE | Description | Exploitability | Impact | Status | Recommended Fix |
|----|-----------|-----|-------------|---------------|--------|--------|----------------|
| SEC-009 | D1 | CWE-125 | **NEON `debug_assert_eq` instead of `assert_eq`**: All 10 aarch64 NEON SIMD functions in `simd_intrinsics.rs` use `debug_assert_eq!` for length validation, stripped in release builds. Mismatched-length slices cause out-of-bounds reads on ARM64. x86_64 paths correctly use `assert_eq!`. | LOW -- Requires a caller to pass mismatched-length slices. Public API accepts arbitrary `&[f32]`. | HIGH -- Buffer overread on ARM64 production servers and Apple Silicon. Undefined behavior; may silently return wrong results or crash. | Confirmed | Change all `debug_assert_eq!` to `assert_eq!` in NEON functions. Cost: one comparison per call (negligible vs. SIMD loop). |
| SEC-010 | D1 | CWE-190 | **Integer overflow in `SoAVectorStorage::grow()`**: `self.capacity * 2` has no `checked_mul`. On 32-bit targets, capacity > 2^31 silently wraps. `grow()` also does not enforce `MAX_CAPACITY` limit. | LOW -- Only exploitable on 32-bit targets with extreme capacity. | HIGH on 32-bit -- Silent integer wrap leads to undersized allocation, subsequent writes corrupt heap. MEDIUM on 64-bit (theoretical only). | Confirmed | Use `checked_mul(2).expect("overflow")` and add `assert!(new_capacity <= MAX_CAPACITY)`. |
| SEC-011 | D4 | CWE-89 | **SQL injection in `ruvector_hnsw_debug`**: Uses manual `replace('\'', "''")` escaping instead of parameterized queries. Bypassable via PostgreSQL Unicode escapes or `E''` strings depending on `standard_conforming_strings` setting. | MEDIUM -- Requires `standard_conforming_strings = off` or Unicode escape injection. Function is labeled "debug" but is a `#[pg_extern]` callable by any SQL user. | HIGH -- Same as SEC-001 but with partial mitigation from quote-doubling. | Confirmed | Replace with `Spi::get_one_with_args` using `$1` parameters. |
| SEC-012 | D4 | CWE-328 | **SipHash-2-4 used for proof attestation**: `proof_store.rs` uses `DefaultHasher` (SipHash) to generate attestation hashes. SipHash is not collision-resistant. Hash seed changes between process restarts, making attestations non-reproducible. | LOW -- Requires the ability to craft collision inputs for attestation data. | HIGH -- If attestations are used for trust/integrity decisions, an attacker can forge matching hashes. Attestation integrity is compromised. | Confirmed | Replace with `sha2::Sha256` (already a dependency in `ruvector-snapshot`). |
| SEC-013 | D4 | CWE-942 | **CORS `allow_origin(Any)` on ruvector-server**: Combined with SEC-002 (no auth), any website can make cross-origin requests to read/write/delete vector data from a user's browser. | HIGH -- Any malicious webpage visited by someone on the same network as the server. | HIGH -- Data exfiltration and manipulation via cross-origin requests. | Confirmed | Restrict to specific origins. At minimum, use `allow_origin(HeaderValue::from_static("..."))`. |
| SEC-014 | D8 | CWE-942 | **CORS `allow_origin(Any)` on ruvllm serve**: The LLM inference server uses the same permissive CORS pattern as ruvector-server. | HIGH -- Same as SEC-013 but for the LLM inference endpoint. | HIGH -- Malicious websites can invoke LLM inference, potentially exfiltrating model outputs or causing resource exhaustion. | Confirmed | Restrict CORS to configured origins. |
| SEC-015 | D5, D6 | CWE-125 | **Systemic `debug_assert_eq` in CNN SIMD**: All SIMD functions in `ruvector-cnn` (D5) and `ruvllm-wasm` (D6) use `debug_assert_eq!` for bounds checking, stripped in release builds. Same pattern as SEC-009 but across CNN and WASM crates. | LOW -- Requires mismatched-length inputs from callers. | HIGH -- Out-of-bounds read in release builds. In WASM context, reads garbage from linear memory. In native context, undefined behavior. | Confirmed | Add `assert_eq!` in safe dispatch wrappers (`dot_product_simd`, `relu_simd`, etc.). |
| SEC-016 | D4 | CWE-20 | **`read_metadata` trusts page contents without magic validation**: Reads `HnswMetaPage` from buffer page without checking `magic` or `version` fields. IVFFlat correctly validates magic; HNSW does not. | LOW -- Requires corrupted pages (disk failure, concurrent modification). | HIGH -- Reading garbage metadata leads to incorrect index behavior. Could cascade into buffer overflows if corrupted dimension count is used. | Confirmed | Add `if meta.magic != HNSW_MAGIC { pgrx::error!(...) }` after reading. |
| SEC-017 | D4 | CWE-125 | **`hnsw_insert` varlena fallback lacks size validation**: Same pattern as SEC-006 but in the INSERT path. No validation that varlena size matches claimed dimensions. | LOW -- Requires corrupted datum passed to INSERT. | HIGH -- Same overread risk as SEC-006. | Confirmed | Add varlena size validation: `if 4 + dims * 4 > data_size { return; }`. |
| SEC-018 | D9 | CWE-287 | **MCP `RVF_KERNEL_SECRET` defaults to random UUID per restart**: When env var is not set, HMAC signing between kernel and bridge is effectively disabled. Bridge cannot verify signatures since it does not know the random secret. | LOW -- Mitigated by Docker internal network isolation. If bridge is ever exposed externally, becomes CRITICAL. | HIGH -- Unauthenticated tool execution through the MCP bridge. | Confirmed | Log a warning when secret is unset. Fail startup if bridge URL is not a Docker-internal address. |
| SEC-019 | D5 | CWE-369 | **Division-by-zero in MLA softmax**: `softmax_inplace()` divides by `sum` which is 0.0 when all scores are `NEG_INFINITY` (fully masked). Produces NaN that propagates through attention computation. | LOW -- Requires all attention keys to be masked out. Edge case but reachable in causal attention with position 0. | HIGH -- NaN propagation corrupts model outputs silently. Downstream consumers receive garbage attention weights. | Confirmed | Guard: `let sum = sum.max(f32::MIN_POSITIVE);` or return uniform distribution when `sum == 0.0`. |
| SEC-020 | D5 | CWE-190 | **Conv2d `output_shape` integer underflow**: `(in_h + 2*padding - kernel_size) / stride + 1` underflows if `in_h + 2*padding < kernel_size` (usize wraps to huge number). Same pattern at 4 call sites. | LOW -- Requires pathological configuration (large kernel, small input). | HIGH -- Massive memory allocation (gigabytes) from wrapped-around output dimensions. OOM crash or excessive allocation. | Confirmed | Add: `if in_h + 2*self.padding < self.kernel_size { return Err(...); }`. |
| SEC-021 | D4 | -- | **No WAL logging for HNSW/IVFFlat page modifications**: Zero calls to `XLogInsert` or WAL-related APIs. Index changes are not crash-safe, not replicated, and not recoverable via PITR. | N/A -- Design limitation, not an exploit. | HIGH -- Data integrity loss after any PostgreSQL crash. Index must be rebuilt with `REINDEX`. Streaming replication does not replicate index state. | Confirmed | Implement WAL logging (complex) or prominently document the limitation and trigger automatic `REINDEX` after recovery. |
| SEC-022 | D5 | CWE-1078 | **Multi-head attention mask parameter silently ignored**: `compute_with_mask()` accepts a mask parameter prefixed with `_mask` and delegates to `compute()`, ignoring it entirely. | N/A -- Correctness bug, not exploitable. | HIGH -- Any caller expecting masking behavior gets unmasked results. In a safety-critical ML pipeline, this could produce incorrect decisions. | Confirmed | Implement actual masking logic or remove the mask parameter and document that masking is unsupported. |

### MEDIUM Findings

| ID | Domain(s) | CWE | Description | Exploitability | Impact | Status | Recommended Fix |
|----|-----------|-----|-------------|---------------|--------|--------|----------------|
| SEC-023 | D1 | CWE-682 | **Hamming distance u8 accumulator overflow in NEON**: The NEON hamming distance implementation accumulates XOR popcounts in a u8 register. For vectors with >255 set-bit differences (>255 bytes with at least one differing bit), the accumulator silently wraps. | LOW | MEDIUM -- Silently wrong distance values for large vectors. | Confirmed | Use u16 or u32 accumulator, or periodically drain to a wider register. |
| SEC-024 | D4 | CWE-401 | **`pg_detoast_datum` memory leak**: Detoasted copies at 4 call sites in `hnsw_am.rs` are never explicitly freed with `pfree()`. PostgreSQL reclaims at transaction end, but bulk COPY accumulates leaks. | N/A | MEDIUM -- Memory pressure during bulk inserts. Not exploitable but degrades performance. | Confirmed | Call `pfree()` on detoasted datum after use, or switch to a per-tuple memory context. |
| SEC-025 | D4 | CWE-841 | **Tenant admin wildcard policy**: Setting `ruvector.tenant_id = '*'` bypasses RLS tenant isolation. Any user with `SET` privileges can bypass row-level security. | MEDIUM -- Requires `SET` privileges on the database. | MEDIUM -- Tenant data isolation breach. Cross-tenant data access. | Confirmed | Restrict the wildcard policy to a specific admin role: `USING (current_user = 'ruvector_admin' AND ...)`. |
| SEC-026 | D8 | CWE-200 | **`psql` invoked with env-var URL in process args**: `hooks.rs:1142` passes the database connection URL (which may contain credentials) as a command-line argument to `psql`. Visible in `/proc/PID/cmdline` to all users on the system. | LOW -- Requires local access to read process arguments. | MEDIUM -- Credential exposure via process listing. | Confirmed | Use `PGPASSWORD` environment variable or `.pgpass` file instead of embedding credentials in the connection URL argument. |
| SEC-027 | D9 | CWE-1021 | **CSP limited to `frame-ancestors` only**: No `script-src`, `style-src`, `connect-src`, or `default-src` directives. Inline scripts allowed. No exfiltration protection. | MEDIUM -- Amplifies impact of any XSS (SEC-008). | MEDIUM -- Without CSP script restrictions, successful XSS has no secondary defense barrier. | Confirmed | Add baseline CSP: `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https:;` |
| SEC-028 | D9 | CWE-352 | **`trustedOrigins: ["*"]` disables SvelteKit built-in CSRF**: Custom CSRF middleware covers form submissions, but any future endpoint bypassing hooks middleware lacks protection. | LOW -- Requires a new endpoint that bypasses the hooks middleware. | MEDIUM -- CSRF on unprotected endpoints. | Confirmed | Narrow `trustedOrigins` to specific known origins, or add explicit documentation of the risk. |
| SEC-029 | D4 | CWE-682 | **NaN handling in HNSW heap ordering**: `partial_cmp(...).unwrap_or(Ordering::Equal)` treats NaN distances as equal to everything, corrupting binary heap ordering. | LOW -- Requires NaN in distance computation (zero-norm cosine edge case). | MEDIUM -- Incorrect search results returned to users. Heap invariant violation. | Confirmed | Filter NaN before insertion: `if distance.is_nan() { continue; }` or use `total_cmp()`. |
| SEC-030 | D5 | CWE-682 | **GNN cosine similarity mixed-precision**: Dot product in f32 but norms in f64. Can produce `cosine_similarity > 1.0` or negative values for near-identical vectors due to precision mismatch. | N/A -- Correctness issue. | MEDIUM -- Silently wrong similarity scores in GNN search. | Confirmed | Compute dot product in f64 as well, or use Kahan summation for f32 path. |
| SEC-031 | D4 | -- | **No input size limits on ruvector-server requests**: `SearchRequest` accepts unbounded `vector: Vec<f32>` and `k` with no upper limit. | HIGH -- Any network client (trivial with SEC-002 unfixed). | MEDIUM -- Resource exhaustion via huge vectors or k=MAX_INT. OOM crash. | Confirmed | Add dimension validation against collection config and `k` upper bound (e.g., 10,000). |
| SEC-032 | D4 | CWE-311 | **No snapshot transfer encryption**: Snapshots stored as unencrypted gzip-compressed bincode. No encryption-at-rest support. | LOW -- Requires filesystem access. | MEDIUM -- Data exposure on shared storage. | Suspected | Add optional AES-256-GCM encryption layer before compression. |

### LOW Findings

| ID | Domain(s) | CWE | Description | Exploitability | Impact | Status | Recommended Fix |
|----|-----------|-----|-------------|---------------|--------|--------|----------------|
| SEC-033 | D4 | CWE-676 | **No `// SAFETY:` comments on 78+ unsafe blocks in D4 `simd.rs`**: Violates Rust best practices (`clippy::undocumented_unsafe_blocks`). No local safety justification at call sites. | N/A -- Code quality issue, not exploitable. | LOW -- Increases risk of introducing bugs during maintenance. Hinders code review. | Confirmed | Add `// SAFETY:` comments documenting why each unsafe operation is valid. |
| SEC-034 | D8 | CWE-20 | **No bounds check on CLI `dimensions` argument**: Accepts `--dimensions 0` without error. Creates a database that fails on subsequent operations with confusing errors. | N/A -- Local CLI, user confusion only. | LOW -- No data corruption, just poor UX. | Confirmed | Add `value_parser = clap::value_parser!(usize).range(1..)`. |
| SEC-035 | D9 | CWE-1004 | **`sameSite="none"` in production cookies**: Required for HuggingFace iframe embedding but weakens CSRF protection for direct deployments. | LOW -- Only relevant for self-hosted instances. | LOW -- Weakened CSRF protection when not using iframe embedding. | Confirmed | Document that self-hosted deployments should set `COOKIE_SAMESITE=lax`. |
| SEC-036 | D4 | -- | **Relaxed atomic ordering for statistics counters**: `AtomicOrdering::Relaxed` on search/insert counters. | N/A | LOW -- Statistics may be slightly inaccurate under high concurrency. No correctness impact. | Confirmed | Acceptable. Document that stats are approximate. |
| SEC-037 | D4 | CWE-667 | **Stale in-memory cache in ruvector-server**: No cache invalidation mechanism. Concurrent modifications from another client may not be visible. | LOW -- Requires multi-client concurrent access. | LOW -- Stale search results. | Suspected | Implement cache invalidation on write operations, or document eventual consistency. |

---

## 2. Attack Surface Mapping

### 2.1 ruvector-server HTTP Endpoints

| Aspect | Assessment | Risk |
|--------|-----------|------|
| **Authentication** | NONE -- zero auth mechanisms | CRITICAL (SEC-002) |
| **Authorization** | NONE -- no RBAC or ownership model | CRITICAL |
| **Input validation** | Minimal -- dimensions checked by core, but no request size limits | HIGH (SEC-031) |
| **Rate limiting** | NONE -- no middleware | HIGH |
| **CORS** | `allow_origin(Any)` + `allow_methods(Any)` + `allow_headers(Any)` | HIGH (SEC-013) |
| **TLS** | NONE -- plain TCP listener | HIGH |
| **Error handling** | Returns internal error messages to client | MEDIUM |
| **Test coverage** | ZERO tests | CRITICAL gap |

**Verdict**: This server MUST NOT be deployed on any network accessible by untrusted clients without implementing authentication, TLS, CORS restrictions, input validation, and rate limiting.

### 2.2 ruvllm serve (LLM Inference Server)

| Aspect | Assessment | Risk |
|--------|-----------|------|
| **Authentication** | NONE | HIGH |
| **CORS** | `allow_origin(Any)` | HIGH (SEC-014) |
| **Input validation** | Model name validated; prompt size unchecked | MEDIUM |
| **Rate limiting** | NONE | MEDIUM |
| **Model safety** | No content filtering, guardrails, or output sanitization | MEDIUM |
| **Resource limits** | No per-request memory or compute bounds | MEDIUM |

**Verdict**: Similar to ruvector-server. Must not be exposed without authentication and CORS restrictions.

### 2.3 WASM Bindings (D6, 33 crates)

| Threat | Assessment | Risk |
|--------|-----------|------|
| **Malicious JS calling WASM** | WASM linear memory is sandboxed. Buffer overflows within WASM cannot escape the sandbox. However, `debug_assert` bugs (SEC-015) cause wrong results, not memory escapes. | LOW |
| **Memory exhaustion** | `BufferPool.release()` is a no-op, causing unbounded memory growth. Malicious JS can trigger this. | MEDIUM |
| **Static mut globals** | `micro-hnsw-wasm` uses 60+ `static mut` globals. Safe in single-threaded WASM but UB if shared across Web Workers. | LOW (single-threaded WASM is the norm) |
| **Stub functions** | 4 CRITICAL stubs return empty results silently. Not a security issue but a correctness/reliability concern. | MEDIUM (reliability) |
| **Type safety** | 30+ TypeScript parameters typed as `any`. Malformed input could trigger panics mapped to JS exceptions. | LOW |

**Verdict**: WASM sandboxing provides strong isolation. Primary risks are correctness (stubs, debug_assert) and resource exhaustion (BufferPool leak).

### 2.4 Node.js Bindings (D7, 10 crates)

| Threat | Assessment | Risk |
|--------|-----------|------|
| **Malicious npm code** | NAPI-RS bindings have zero `unsafe` in 9/10 crates. Type conversions go through `serde-wasm-bindgen` or NAPI-RS safe wrappers. | LOW |
| **Lock poisoning** | 37 `unwrap()` calls on `RwLock` across binding crates. If any native code panics, subsequent calls will also panic (cascading failure). | MEDIUM |
| **Memory leaks** | No memory leak tests exist. Long-running Node.js processes with native bindings may accumulate leaked allocations. | MEDIUM |
| **Thread safety** | `Arc<RwLock>` patterns used correctly. `parking_lot::RwLock` used in some crates (does not poison). | LOW |

**Verdict**: Good isolation through NAPI-RS safe wrappers. Main risks are lock poisoning and memory leaks in long-running processes.

### 2.5 CLI Tools (D8)

| Threat | Assessment | Risk |
|--------|-----------|------|
| **Command injection** | `Command::new` uses `.arg()` API (not shell interpolation). No injection risk. | NONE |
| **Path traversal** | MCP handler has excellent path validation. Direct CLI commands have no path validation (SEC-034). | LOW (local tool) |
| **Credential exposure** | `psql` invoked with credentials in process args (SEC-026). | MEDIUM |
| **Input validation** | `dimensions=0` accepted, unknown metric silently defaults to Cosine. | LOW |

**Verdict**: Acceptable risk for a local CLI tool. The `psql` credential exposure should be fixed.

### 2.6 PostgreSQL Extension (D4)

| Threat | Assessment | Risk |
|--------|-----------|------|
| **SQL injection** | 8 functions with direct string interpolation (SEC-001, SEC-011). Graph/SPARQL and tenancy modules are properly parameterized. | CRITICAL |
| **Buffer overflow** | Page boundary overflows in HNSW write operations (SEC-004, SEC-005). | CRITICAL |
| **Varlena overread** | Missing size validation in 3 varlena parsing paths (SEC-006, SEC-017). | HIGH |
| **Crash safety** | No WAL logging -- index lost on crash (SEC-021). | HIGH |
| **Tenant isolation** | RLS bypass via wildcard `*` tenant ID (SEC-025). | MEDIUM |
| **Memory safety** | `debug_assert`-only SIMD checks in release builds (SEC-007). | CRITICAL |

**Verdict**: The PostgreSQL extension is the highest-risk component. A memory safety bug here crashes the entire PostgreSQL server, potentially corrupting all databases. The SQL injection and page overflow findings must be fixed before any deployment.

### 2.7 SvelteKit UI (D9)

| Threat | Assessment | Risk |
|--------|-----------|------|
| **XSS** | 1 path without DOMPurify (SEC-008). 3 paths correctly sanitized. | HIGH |
| **CSRF** | Custom middleware covers forms. Built-in SvelteKit CSRF disabled (SEC-028). | MEDIUM |
| **Authentication** | Solid OIDC+PKCE implementation with email verification and allowlists. | LOW |
| **Cookie security** | `httpOnly`, `secure`, configurable `sameSite`. | LOW |
| **CSP** | Only `frame-ancestors` set (SEC-027). No script-src protection. | MEDIUM |
| **MCP bridge auth** | Secret defaults to random UUID when unset (SEC-018). | HIGH |

**Verdict**: Authentication is production-grade. XSS and CSP gaps need remediation before handling sensitive conversations.

---

## 3. Supply Chain Assessment

### 3.1 Rust Dependencies (cargo audit)

**Total dependencies in `Cargo.lock`**: 1,241
**Vulnerabilities found**: 6
**Allowed warnings**: 17 (including 2 unsoundness advisories)

| CVE/Advisory | Package | Version | CVSS | Fix Available | Impact Chain |
|--------------|---------|---------|------|---------------|-------------|
| RUSTSEC-2026-0041 | `lz4_flex` | 0.11.5 | **8.2 HIGH** | Yes (>=0.11.6) | ruvector-delta-core -> delta-wasm, delta-index, delta-graph, delta-consensus, mcp-brain-server |
| RUSTSEC-2026-0037 | `quinn-proto` | 0.11.13 | **8.7 HIGH** | Yes (>=0.11.14) | reqwest -> rvagent-*, mcp-brain-server, ruvector-scipix, fastembed |
| RUSTSEC-2023-0071 | `rsa` | 0.9.10 | **5.9 MEDIUM** | **No fix available** | sqlx-mysql -> prime-radiant |
| RUSTSEC-2024-0421 | `idna` | 0.5.0 | Unscored | Yes (>=1.0.0) | URL parsing in networking crates |
| RUSTSEC-2024-0437 | `protobuf` | 2.28.0 | Unscored | Yes (>=3.7.2) | Protobuf message handling (DoS via recursion) |
| RUSTSEC-2026-0049 | `rustls-webpki` | 0.103.9 | Unscored | Yes (>=0.103.10) | TLS certificate validation |

**Unsoundness Warnings (allowed but notable)**:

| Advisory | Package | Issue |
|----------|---------|-------|
| RUSTSEC-2026-0002 | `lru` 0.12.5 | `IterMut` violates Stacked Borrows by invalidating internal pointer. Undefined behavior under Miri. |
| RUSTSEC-2024-0408 | `pprof` 0.13.0 | Unsound usages of `std::slice::from_raw_parts`. |

**Assessment**: The `lz4_flex` vulnerability is most concerning because it affects data decompression in the delta-core module (used for snapshot/delta operations). An attacker who can supply crafted compressed data could leak uninitialized memory. The `quinn-proto` DoS affects all networking code via reqwest. The `rsa` timing side-channel has no fix and affects MySQL connections only (low exposure).

### 3.2 NPM Dependencies

**Total NPM packages in monorepo**: 56
**Key security-sensitive packages**:

| Package | Status | Concern |
|---------|--------|---------|
| `dompurify` / `isomorphic-dompurify` | v3.2.4 / 2.13.0 | Current. `isomorphic-dompurify` pinned to exact version (misses security patches). |
| `mongodb` | v5.x | One major version behind (v6.x current). May miss security fixes. |
| `openid-client` | v5.x | One major version behind (v6.x current). Security-sensitive library. |
| `marked` | v12.x | Current. |
| `openai` | v4.x | Current. |

**Lockfiles**:

| File | Present | Committed | Assessment |
|------|---------|-----------|-----------|
| `Cargo.lock` | Yes | Yes | GOOD -- deterministic Rust builds |
| `package-lock.json` (root) | Yes | Yes | GOOD |
| `ui/ruvocal/package-lock.json` | Yes | Yes | GOOD |

All lockfiles are present and committed, ensuring reproducible builds.

### 3.3 Supply Chain Risk Summary

- **Rust**: 6 CVEs, 2 unsoundness warnings. 4 of 6 CVEs have available fixes (upgradeable). The `rsa` CVE has no fix.
- **NPM**: No known CVEs flagged, but 2 security-sensitive packages are one major version behind.
- **Lockfiles**: All present and committed. Reproducible builds ensured.
- **No malicious packages detected** in either ecosystem.

---

## 4. OWASP Top 10 2021 Coverage

| Category | Findings | Status |
|----------|----------|--------|
| **A01 Broken Access Control** | SEC-002 (no auth), SEC-003 (path traversal), SEC-025 (tenant bypass) | FAIL |
| **A02 Cryptographic Failures** | SEC-012 (SipHash for attestation), SEC-032 (no encryption at rest) | FAIL |
| **A03 Injection** | SEC-001 (SQLi), SEC-008 (XSS), SEC-011 (SQLi), SEC-004/005/006/007 (memory corruption via crafted input) | FAIL |
| **A04 Insecure Design** | SEC-021 (no WAL logging), SEC-022 (ignored mask parameter) | FAIL |
| **A05 Security Misconfiguration** | SEC-013/014 (CORS wildcard), SEC-027 (weak CSP), SEC-028 (CSRF disabled) | FAIL |
| **A06 Vulnerable Components** | 6 CVEs in cargo audit, 2 outdated NPM security libraries | FAIL |
| **A07 Auth Failures** | SEC-002 (no auth), SEC-018 (MCP secret fallback) | FAIL |
| **A08 Software Integrity** | Lockfiles committed (PASS). No code signing, no SBOM generation (gaps). | PARTIAL |
| **A09 Logging Failures** | No security event logging in ruvector-server. No audit trail for vector operations. | FAIL |
| **A10 SSRF** | No SSRF vectors identified in current code. URL validation present in MCP handler. | PASS |

**OWASP Score: 1/10 PASS, 1/10 PARTIAL, 8/10 FAIL**

---

## 5. Systemic Issues

### 5.1 The `debug_assert` Pattern (~95 sites)

The most pervasive security-relevant code pattern in the monorepo is the use of `debug_assert!` / `debug_assert_eq!` for safety-critical precondition checks in `unsafe` code. This pattern appears in:

- **D1** `simd_intrinsics.rs`: 10 NEON functions (SEC-009)
- **D4** `distance/simd.rs`: 14 pointer functions + 6 batch functions (SEC-007)
- **D4** `distance/simd.rs`: 4 NEON slice functions
- **D5** `ruvector-cnn/simd/`: All SIMD dispatch functions (SEC-015)
- **D6** `ruvllm-wasm`: SIMD code (same pattern)

In release builds (`--release`, which is the standard for production), all `debug_assert` calls are compiled away to nothing. This means:
- Null pointer dereferences become undefined behavior instead of panics
- Mismatched-length buffer reads become silent out-of-bounds access
- Batch result buffer overflows become silent heap corruption

**Total estimated affected sites**: ~95 across the codebase.

**Root cause**: A convention to treat `debug_assert` as "sufficient for unsafe code" when it is fundamentally insufficient. The performance cost of a single comparison or null check is negligible compared to the SIMD loop body.

**Fix**: Global search-and-replace of `debug_assert` to `assert` in all `unsafe fn` contexts, or add explicit `if ... return` guards.

### 5.2 Missing `// SAFETY:` Documentation

Across D1, D4, and D5, approximately 300+ `unsafe` blocks lack `// SAFETY:` comments. While this is a code quality issue rather than a vulnerability, it directly increases the probability of introducing bugs during maintenance. The Rust ecosystem standard (`clippy::undocumented_unsafe_blocks`) exists specifically because undocumented unsafe code is the primary source of soundness bugs.

### 5.3 Zero Test Coverage on Security-Critical Paths

| Component | Security Test Coverage | Gap |
|-----------|----------------------|-----|
| ruvector-server | 0 tests | No auth, CORS, input validation, or error handling tests |
| SQL injection paths | Benign inputs only | No injection payload tests |
| Path traversal | Tested in MCP handler | Not tested in snapshot storage or CLI |
| NEON SIMD | Only on aarch64 CI | x86_64 CI never tests NEON paths |
| Page boundary checks | Tested (added post-issue-164) | Added reactively, not proactively |

---

## 6. Recommendations by Priority

### IMMEDIATE (Must fix before any deployment)

| # | Finding(s) | Action | Effort |
|---|-----------|--------|--------|
| 1 | SEC-001, SEC-011 | Fix SQL injection in DAG functions and debug function. Use `pg_parse_query()` validation or `Spi::run_with_args` parameterized queries. | 1-2 days |
| 2 | SEC-002, SEC-013, SEC-031 | Add API key authentication, restrict CORS, add input size limits to ruvector-server. | 2-3 days |
| 3 | SEC-004, SEC-005 | Add page boundary size checks before all HNSW page writes. Reject vectors that exceed `(BLCKSZ - header) / 4` dimensions, or implement multi-page storage. | 1-2 days |
| 4 | SEC-007, SEC-009, SEC-015 | Global replacement of `debug_assert` with `assert` in all unsafe SIMD functions across D1, D4, D5, D6. | 1 day |
| 5 | SEC-003 | Sanitize snapshot IDs: validate against `[a-zA-Z0-9_-]+` and canonicalize paths. | 0.5 days |
| 6 | SEC-014 | Restrict CORS on ruvllm serve to configured origins. | 0.5 days |

### SHORT-TERM (Fix within 30 days)

| # | Finding(s) | Action | Effort |
|---|-----------|--------|--------|
| 7 | SEC-006, SEC-016, SEC-017 | Add varlena size validation and magic number checks in all HNSW/IVFFlat varlena parsing paths. | 1-2 days |
| 8 | SEC-008 | Add `DOMPurify.sanitize()` to MarkdownBlock.svelte `{@html}` output. | 0.5 days |
| 9 | SEC-027 | Implement baseline Content Security Policy with `script-src`, `style-src`, `connect-src`, `default-src`. | 1 day |
| 10 | SEC-012 | Replace `SipHash` with `sha2::Sha256` in proof attestation hashing. | 0.5 days |
| 11 | SEC-018 | Add startup warning/failure when `RVF_KERNEL_SECRET` is unset and bridge URL is non-local. | 0.5 days |
| 12 | SEC-019, SEC-020, SEC-022 | Fix MLA softmax division-by-zero, Conv2d underflow, and multi-head attention mask bypass. | 1-2 days |
| 13 | Supply chain | Upgrade `lz4_flex` (>=0.11.6), `quinn-proto` (>=0.11.14), `protobuf` (>=3.7.2), `rustls-webpki` (>=0.103.10), `idna` (>=1.0.0). | 1 day |
| 14 | Supply chain | Update `mongodb` to v6.x and `openid-client` to v6.x in ui/ruvocal. | 1-2 days |

### MEDIUM-TERM (Fix within 90 days)

| # | Finding(s) | Action | Effort |
|---|-----------|--------|--------|
| 15 | SEC-021 | Implement WAL logging for HNSW and IVFFlat page modifications, or document the limitation prominently and implement automatic `REINDEX` on recovery. | 2-4 weeks |
| 16 | SEC-024, SEC-029 | Fix pg_detoast memory leaks, NaN handling in heap ordering, and tenant admin wildcard restriction. | 2-3 days |
| 17 | SEC-026 | Remove credentials from `psql` command-line arguments. Use `PGPASSWORD` env var or `.pgpass`. | 0.5 days |
| 18 | SEC-023, SEC-030 | Fix hamming accumulator overflow and GNN mixed-precision cosine similarity. | 1-2 days |
| 19 | Systemic | Add `// SAFETY:` comments to all 300+ undocumented unsafe blocks. Enable `clippy::undocumented_unsafe_blocks` lint. | 3-5 days |
| 20 | Testing | Write security-focused tests: SQL injection payloads, path traversal attempts, boundary-condition vectors, NEON cross-compilation tests. | 1-2 weeks |
| 21 | Testing | Add integration tests for ruvector-server (currently zero tests). | 1 week |

### LONG-TERM (Architectural improvements)

| # | Action | Rationale |
|---|--------|-----------|
| 22 | Implement TLS support in ruvector-server and ruvllm serve | All data currently transmitted in plaintext |
| 23 | Add RBAC (Role-Based Access Control) to ruvector-server | Currently all-or-nothing access |
| 24 | Implement rate limiting middleware | No protection against resource exhaustion |
| 25 | Add security event logging and audit trail | Currently no security observability |
| 26 | Generate SBOM (Software Bill of Materials) for all artifacts | Required for supply chain transparency |
| 27 | Implement multi-page vector storage in PostgreSQL extension | Current single-page design limits dimensions to ~2,034 |
| 28 | Add automated security scanning to CI/CD pipeline | Currently no security gates in CI |
| 29 | Consider `unsafe_op_in_unsafe_fn` lint globally | Currently only enabled in `ruvector-gnn`. Would require explicit `unsafe` blocks within unsafe functions. |
| 30 | Add fuzz testing for SIMD functions and varlena parsing | High-value targets for automated bug finding |

---

## 7. Risk Matrix

```
                    IMPACT
              Low    Med    High   Critical
         +------+------+------+---------+
  High   |      |SEC-31|SEC-13|SEC-001  |
         |      |      |SEC-14|SEC-002  |
  E      +------+------+------+---------+
  X Med  |      |SEC-25|SEC-08|SEC-004  |
  P      |      |      |SEC-18|SEC-005  |
  L      |      |      |SEC-20|SEC-007  |
  O      +------+------+------+---------+
  I Low  |SEC-33|SEC-23|SEC-09|SEC-006  |
  T      |SEC-34|SEC-26|SEC-10|         |
  A      |SEC-35|SEC-29|SEC-12|         |
  B      |SEC-36|SEC-30|SEC-15|         |
  I      |SEC-37|SEC-32|SEC-16|         |
  L      |      |      |SEC-17|         |
  I      |      |      |SEC-19|         |
  T      |      |      |SEC-21|         |
  Y      |      |      |SEC-22|         |
         +------+------+------+---------+
```

---

## 8. Conclusion

The RuVector monorepo has significant security debt concentrated in Domain 4 (PostgreSQL extension and vector server). The 8 CRITICAL findings represent genuine exploitation risk in any deployment scenario. The systemic `debug_assert` pattern (~95 sites) is the single highest-leverage fix: resolving it addresses memory safety across 4 domains in a single mechanical change.

Positive observations:
- Zero real secrets committed to the repository (Phase 1 PASS)
- Excellent SQL injection prevention in the tenancy and graph/SPARQL modules (should be the template for DAG functions)
- Strong OIDC authentication in the UI layer
- Zero unsafe code in D3 (Distributed), D8 (CLI), D9 (UI)
- All lockfiles present and committed
- WASM sandboxing provides strong isolation for browser-side code

The immediate priority is clear: fix SQL injection, add authentication to the HTTP server, add page boundary checks in the PostgreSQL extension, and globally replace `debug_assert` with `assert` in unsafe code. These 6 actions address all 8 CRITICAL findings and can be completed in approximately 1 week of focused effort.

---

*Report generated by QE Security Auditor v3 -- Phase 3 Consolidated Security Posture Assessment*
*Based on Phase 1 automated scans and Phase 2 deep domain audits (2026-03-29)*
