# Phase 3: Test Gap Analysis & Coverage Mapping

**Date**: 2026-03-29
**Scope**: All 10 domains of the RuVector monorepo
**Methodology**: Automated grep/glob/LOC analysis with manual test quality sampling

---

## 1. Executive Summary

The RuVector monorepo contains **113 Rust crates** with **896,197 lines of production code** and **15,857 `#[test]` functions** across 1,632 files with inline test modules and 265 dedicated test files. An additional 81 JS/TS test files cover the NPM/UI layer.

**Key findings**:
- **31 Rust crates have zero tests** (16,290 LOC exposed)
- **16 WASM binding crates have zero tests** (10,270 LOC)
- **8 NAPI-RS Node binding crates have zero tests** (6,020 LOC)
- **18 NPM packages have zero JS/TS tests** (63,839 LOC)
- **D3 Distributed Systems has zero integration tests** -- the highest-risk gap
- **D4 ruvector-server has zero tests** for the primary API surface
- Test-to-code ratios vary wildly: D8 ruvllm at 1:5.7 (good) vs D2 ruvector-postgres at 1:7.1 (sparse for 65K LOC)

---

## 2. Test Inventory by Domain

### D1: Vector Core

| Crate | Prod LOC | Pub Fns | #[test] | Inline Files | Test Files | Bench Files |
|-------|----------|---------|---------|--------------|------------|-------------|
| ruvector-core | 17,248 | 364 | 433 | 30 | 11 | 8 |
| ruvector-math | 13,166 | 403 | 148 | 39 | 0 | 5 |
| ruvector-filter | 1,507 | 36 | 16 | 4 | 0 | 0 |
| ruvector-hyperbolic-hnsw | 2,452 | 85 | 45 | 5 | 1 | 1 |
| micro-hnsw-wasm | 1,262 | 0 | **0** | 0 | 0 | 0 |
| **TOTAL** | **35,635** | **888** | **642** | **78** | **12** | **14** |

- **Test LOC**: ~5,722 (dedicated) + ~6,571 (inline) = ~12,293
- **Test:Code ratio**: 1:2.9 (line-based); 0.72 tests per pub fn
- **Gaps**: micro-hnsw-wasm entirely untested (1,262 LOC). No NaN input tests found for distance functions. HNSW delete has some coverage (test_delete_nonexistent, test_delete_updates_len) but no stress/concurrent-delete tests. Concurrent access has basic coverage (test_hnsw_parallel_batch_insert, test_soa_concurrent_reads).

### D2: Storage & Persistence

| Crate | Prod LOC | Pub Fns | #[test] | Inline Files | Test Files | Bench Files |
|-------|----------|---------|---------|--------------|------------|-------------|
| ruvector-collections | 930 | 21 | 4 | 2 | 0 | 0 |
| ruvector-postgres | 65,222 | 1,430 | 570 | 89 | 0 | 7 |
| ruvector-snapshot | 844 | 22 | 5 | 4 | 0 | 0 |
| **TOTAL** | **66,996** | **1,473** | **579** | **95** | **0** | **7** |

- **Test LOC**: ~9,538 (inline only, zero dedicated test files)
- **Test:Code ratio**: 1:7.0 (line-based); 0.39 tests per pub fn
- **Gaps**: All 579 tests are inline -- no integration test files exist. ruvector-collections has only 4 tests for 21 pub fns. No ACID/transaction property tests found (Phase 2 noted TODO stubs). Single SQL injection test exists (test_sql_injection_attempts in tenancy/validation.rs). ruvector-snapshot has only 5 tests for snapshot transfer logic.

### D3: Distributed Systems

| Crate | Prod LOC | Pub Fns | #[test] | Inline Files | Test Files | Bench Files |
|-------|----------|---------|---------|--------------|------------|-------------|
| ruvector-raft | 2,171 | 90 | 23 | 5 | 0 | 0 |
| ruvector-replication | 2,097 | 82 | 23 | 6 | 0 | 0 |
| ruvector-cluster | 1,819 | 60 | 12 | 4 | 0 | 0 |
| ruvector-delta-consensus | 1,620 | 58 | 19 | 4 | 0 | 0 |
| **TOTAL** | **7,707** | **290** | **77** | **19** | **0** | **0** |

- **Test LOC**: ~1,341 (inline only)
- **Test:Code ratio**: 1:5.7 (line-based); 0.27 tests per pub fn
- **CRITICAL**: Zero dedicated test files. Zero integration tests. Zero multi-node tests. Zero network partition tests. All 77 tests are single-node unit tests of individual data structures (log append/truncate, snapshot creation). No tests exercise the Raft state machine transitions under concurrent conditions or with simulated network failures.

### D4: API & Server

| Crate | Prod LOC | Pub Fns | #[test] | Inline Files | Test Files | Bench Files |
|-------|----------|---------|---------|--------------|------------|-------------|
| ruvector-server | 555 | 14 | **0** | 0 | 0 | 0 |
| ruvector-cli | 6,679 | 140 | 59 | 6 | 4 | 0 |
| ruvector-router-core | 1,795 | 47 | 17 | 6 | 0 | 0 |
| ruvector-router-cli | 308 | 0 | **0** | 0 | 0 | 0 |
| ruvector-router-ffi | 209 | 8 | **0** | 0 | 0 | 0 |
| **TOTAL** | **9,546** | **209** | **76** | **12** | **4** | **0** |

- **Test LOC**: ~935 (dedicated) + inline
- **Test:Code ratio**: 1:10+ (line-based); 0.36 tests per pub fn
- **CRITICAL**: ruvector-server (the primary HTTP API) has zero tests. Health check, readiness, points, and collections routes are completely untested. ruvector-router-cli and ruvector-router-ffi also have zero tests.

### D5: ML & Neural

| Crate | Prod LOC | Pub Fns | #[test] | Inline Files | Test Files | Bench Files |
|-------|----------|---------|---------|--------------|------------|-------------|
| prime-radiant | 52,466 | 1,606 | 713 | 90 | 12 | 13 |
| ruvector-attention | 19,377 | 494 | 269 | 58 | 0 | 2 |
| ruvector-cnn | 17,905 | 397 | 410 | 38 | 9 | 2 |
| ruvector-gnn | 8,527 | 185 | 221 | 13 | 1 | 0 |
| ruvector-coherence | 1,163 | 29 | 24 | 5 | 1 | 0 |
| ruvector-sparse-inference | 8,248 | 229 | 176 | 21 | 8 | 2 |
| **TOTAL** | **107,686** | **2,940** | **1,813** | **225** | **31** | **19** |

- **Test LOC**: ~13,719 (dedicated) + inline
- **Test:Code ratio**: 1:7.8 (line-based); 0.62 tests per pub fn
- **Gaps**: ruvector-attention has 269 tests but zero dedicated test files for 19,377 LOC. Numerical edge case testing is limited -- no explicit tests for adversarial inputs (NaN, Inf, subnormal floats) found in FlashAttention paths. prime-radiant has the best test coverage in this domain but 1,606 pub fns vs 713 tests means many functions are untested.

### D6: Specialized Crates

This domain spans ~30 crates. Key highlights:

| Crate | Prod LOC | #[test] | Status |
|-------|----------|---------|--------|
| ruvector-mincut | 45,911 | 887 | Well-tested |
| ruvector-graph | 16,840 | 369 | Moderate |
| ruqu-core | 26,093 | 792 | Well-tested |
| ruvector-nervous-system | 14,708 | 487 | Good |
| ruvector-mincut-gated-transformer | 20,273 | 412 | Good |
| ruQu | 14,251 | 382 | Good |
| rvlite | 13,085 | 131 | Sparse |
| sona | 10,721 | 87 | Sparse (402 pub fns) |
| ruvector-graph-transformer | 11,625 | 186 | Moderate |
| ruvector-temporal-tensor | 11,446 | 384 | Good |
| ruvector-domain-expansion | 6,465 | 92 | Sparse |
| ruvector-dag | 8,188 | 158 | Moderate |
| cognitum-gate-kernel | 4,536 | 156 | Good |
| cognitum-gate-tilezero | 2,971 | 190 | Good |
| ruvector-robotics | 9,682 | 298 | Good |
| ruvector-fpga-transformer | 7,582 | 64 | Sparse |
| neural-trader-core | 197 | 2 | Minimal |
| neural-trader-coherence | 300 | 7 | Minimal |
| neural-trader-replay | 294 | 3 | Minimal |
| neural-trader-wasm | 895 | 10 | Minimal |

**Sub-crate ecosystems**:
- **ruvix** (22 sub-crates): 70,511 total LOC, 1,438 tests -- well-distributed
- **rvf** (18 sub-crates): 52,668 total LOC, 1,255 tests -- rvf-cli (2,073 LOC), rvf-server (2,363 LOC), rvf-node (1,058 LOC) have zero tests
- **rvAgent** (9 sub-crates): 38,619 total LOC, 1,153 tests -- rvagent-wasm has 61 tests (was reported as zero in Phase 2, corrected here)

### D7: WASM & Node Bindings

**WASM crates with zero Rust tests (16 crates, 10,270 LOC)**:
| Crate | LOC |
|-------|-----|
| ruvector-attention-unified-wasm | 2,598 |
| micro-hnsw-wasm | 1,262 |
| ruvector-graph-wasm | 1,099 |
| ruvector-attention-wasm | 780 |
| ruvector-mincut-wasm | 778 |
| ruvector-hyperbolic-hnsw-wasm | 632 |
| ruvector-math-wasm | 550 |
| ruvector-domain-expansion-wasm | 503 |
| ruvector-mincut-gated-transformer-wasm | 488 |
| ruvector-cnn-wasm | 470 |
| ruvector-gnn-wasm | 410 |
| ruvector-sparsifier-wasm | 235 |
| ruvector-verified-wasm | 250 |
| ruvector-router-wasm | 137 |
| ruvector-fpga-transformer-wasm | 73 |
| ruvector-temporal-tensor-wasm | 5 |

**Node (NAPI-RS) crates with zero tests (8 crates, 6,020 LOC)**:
| Crate | LOC |
|-------|-----|
| ruvector-attention-node | 2,515 |
| ruvector-solver-node | 1,182 |
| ruvector-node | 779 |
| ruvector-mincut-node | 545 |
| ruvector-gnn-node | 421 |
| ruvector-tiny-dancer-node | 286 |
| agentic-robotics-node | 233 |
| ruvector-mincut-brain-node | 59 |

**NPM packages with zero JS/TS tests (18 packages, 63,839 LOC)**:
| Package | LOC |
|---------|-----|
| npm/ruvector | 28,650 |
| npm/postgres-cli | 10,358 |
| npm/ruvector-wasm-unified | 4,394 |
| npm/graph-data-generator | 4,186 |
| npm/cli | 3,472 |
| npm/rvf | 2,619 |
| npm/raft | 1,726 |
| npm/replication | 1,596 |
| npm/rvlite | 1,279 |
| npm/cognitum-gate-wasm | 1,062 |
| npm/ospipe | 971 |
| npm/rvf-mcp-server | 798 |
| npm/rvdna | 663 |
| npm/pi-brain | 640 |
| npm/scipix | 566 |
| npm/ruvllm-wasm | 443 |
| npm/ruvllm-cli | 296 |
| npm/node | 120 |

**NPM packages with tests**: ruvbot (22 tests), agentic-synth (11), agentic-synth-examples (5), ruvector-extensions (4), ruvllm (3), rudag (1), rvf-solver (1), sona (1) = **48 JS/TS test files total**

### D8: LLM & Inference

| Crate | Prod LOC | Pub Fns | #[test] | Inline Files | Test Files | Bench Files |
|-------|----------|---------|---------|--------------|------------|-------------|
| ruvllm | 140,891 | 2,837 | 2,426 | 152 | 28 | 0 |
| ruvllm-cli | 3,751 | 18 | 11 | 7 | 0 | 0 |
| ruvllm-wasm | 10,596 | 376 | 94 | 16 | 2 | 0 |
| **TOTAL** | **155,238** | **3,231** | **2,531** | **175** | **30** | **0** |

- **Test LOC**: ~27,372 (inline) + dedicated test files
- **Test:Code ratio**: 1:5.7 (line-based); 0.78 tests per pub fn
- **Gaps**: ruvllm-cli has only 11 tests for CLI integration. Metal/GPU tests silently skip when hardware unavailable (flaky test indicator). No benchmarks despite being the performance-critical inference engine.

### D9: UI (RuVocal)

| Metric | Value |
|--------|-------|
| Svelte components | 84 |
| Route pages | 13 |
| TS/JS lib modules | 174 |
| Test files | 26 |
| Test LOC | 4,724 |
| Total UI LOC | 39,630 |

- **Test:Code ratio**: 1:8.4
- **Coverage**: 26 test files covering server APIs, tree utilities, migrations, and 1 component test (MarkdownRenderer). The remaining 83 Svelte components have zero component tests. Server API tests are well-structured (conversations, user, misc).

### D10: MCP & Brain + Robotics

| Crate | Prod LOC | Pub Fns | #[test] | Inline Files | Test Files |
|-------|----------|---------|---------|--------------|------------|
| mcp-brain-server | 22,435 | 357 | 146 | 14 | 0 |
| mcp-brain | 2,456 | 64 | 33 | 2 | 0 |
| mcp-gate | 1,337 | 14 | 6 | 3 | 0 |
| agentic-robotics-core | 669 | 28 | 9 | - | - |
| agentic-robotics-rt | 483 | 17 | 6 | - | - |
| agentic-robotics-mcp | 506 | 12 | **0** | - | - |
| agentic-robotics-node | 233 | 13 | **0** | - | - |
| agentic-robotics-embedded | 41 | 0 | 1 | - | - |
| **TOTAL** | **28,160** | **505** | **201** | **19** | **0** |

- **Test:Code ratio**: 1:140+ (line-based for LOC vs dedicated test LOC -- all inline)
- **Gaps**: mcp-brain-server has 146 tests for 22K LOC and 357 pub fns. Zero dedicated integration test files. agentic-robotics-mcp and agentic-robotics-node have zero tests.

---

## 3. Test-to-Code Ratios Summary

| Domain | Prod LOC | Test Fns | Test LOC (est.) | Line Ratio | Fn Ratio (test/pub) |
|--------|----------|----------|-----------------|------------|---------------------|
| D1 Vector Core | 35,635 | 642 | ~12,293 | 1:2.9 | 0.72 |
| D2 Storage | 66,996 | 579 | ~9,538 | 1:7.0 | 0.39 |
| D3 Distributed | 7,707 | 77 | ~1,341 | 1:5.7 | 0.27 |
| D4 API & Server | 9,546 | 76 | ~935 | 1:10.2 | 0.36 |
| D5 ML & Neural | 107,686 | 1,813 | ~13,719 | 1:7.8 | 0.62 |
| D6 Specialized | ~275,000 | ~8,200 | ~55,000 | 1:5.0 | ~0.50 |
| D7 WASM+Node | ~40,000 | ~358 | ~3,000 | 1:13.3 | ~0.10 |
| D8 LLM | 155,238 | 2,531 | ~27,372 | 1:5.7 | 0.78 |
| D9 UI | 39,630 | ~200 | 4,724 | 1:8.4 | ~0.15 |
| D10 MCP & Brain | 28,160 | 201 | ~2,500 | 1:11.3 | 0.40 |
| **TOTAL** | **~896K** | **~15,857** | **~130K** | **1:6.9** | **~0.50** |

**Interpretation**: The overall test-to-code ratio of 1:6.9 is moderate for a systems-level Rust project. However, the distribution is highly uneven -- D1 and D8 are reasonably well-covered while D3, D4, D7, and D10 have severe deficits.

---

## 4. Test Quality Analysis

### 4.1 Sampled Files (20 files across domains)

| # | File | Domain | Quality Assessment |
|---|------|--------|--------------------|
| 1 | prime-radiant/src/security/limits.rs | D5 | **Behavior-focused**. Tests public API contracts (can_add_node, is_valid_state_dim). Missing: boundary at max-1/max/max+1 not systematic. No assertion messages. |
| 2 | ruvector-cnn/src/layers/quantized_conv2d.rs | D5 | **Good structure**. Tests creation and forward pass. Missing: edge cases (zero channels, mismatched dimensions). No assertion messages. |
| 3 | ruvector-collections/src/collection.rs | D2 | **Boundary-aware**. Tests zero dimensions, too-large dimensions. format_bytes covers multiple scales. Only 2 test fns for 21 pub fns -- very sparse. |
| 4 | ruvector-postgres/src/graph/cypher/executor.rs | D2 | **Behavioral**. Tests CREATE and MATCH Cypher queries. Missing: error paths, malformed queries, injection attempts. |
| 5 | ruvector-verified/src/invariants.rs | D6 | **Good assertions**. Uses custom messages ("expected at least 11 builtins, got {}"). Tests deduplication. Clean isolation. |
| 6 | ruvector-delta-graph/src/lib.rs | D6 | **Well-structured**. Tests builder, apply, compose, affected_nodes. Clear test names. Missing: error paths for invalid operations. |
| 7 | ruvector-attention/src/info_geometry/fisher.rs | D5 | **Mathematical correctness**. Tests tangent space constraints, CG solve verification, distance properties (identity, positivity). Good numerical tolerance. |
| 8 | ruvector-mincut-gated-transformer/src/kernel/qgemm.rs | D6 | **Excellent**. Manual expected-value calculations in comments. Tests bias, quantize-dequantize roundtrip. Precision tolerance checks. |
| 9 | ruvector-nervous-system/src/separate/sparsification.rs | D6 | **Comprehensive**. Tests creation, set/check, from_indices (dedup), intersection, union, Jaccard similarity (identical, disjoint, partial), Hamming distance. Edge cases covered. |
| 10 | ruvllm/src/metal/context.rs | D8 | **Platform-conditional**. Silently returns on non-Metal hardware (flaky indicator). Tests flash_attention, rms_norm. Uses assertion messages. |
| 11 | rvf/rvf-crypto/src/hash.rs | D6 | **Excellent**. Tests empty input, determinism, collision resistance, prefix consistency, arbitrary output length, NIST known-answer vector. Textbook test design. |
| 12 | ruvix/crates/fs/src/vfs.rs | D6 | **Good coverage**. Tests InodeId validity, FileType modes, OpenFlags combinations, seek operations (Start/Current/End), Inode creation, OpenFileTable. Good isolation. |
| 13 | ruvector-raft/src/log.rs | D3 | **Structural tests**. append, get, truncate, matches, snapshot_creation, entries_from. All single-node. Missing: concurrent access, term conflict resolution. |
| 14 | ruvector-postgres/src/graph/sparql/results.rs | D2 | **Format coverage**. Tests JSON, XML, CSV, TSV, N-Triples, ASK queries, XML escaping. Good helper fn (create_test_select). Missing: empty results, large result sets. |
| 15 | ruvector-robotics/src/bridge/converters.rs | D6 | **Good structure**. Tests point cloud conversions with error cases (LengthMismatch, EmptyInput). Uses Result-based assertions. |
| 16 | ruvix/crates/cli/src/commands/keys.rs | D6 | **Integration-style**. Uses tempdir for isolation. Tests generate, sign, verify end-to-end. Good real-world coverage. |
| 17 | rvf/rvf-types/src/flags.rs | D6 | **Thorough**. Tests empty, set/check, clear, reserved bits masking, all known flags. Clean bit manipulation coverage. |
| 18 | rvf/rvf-solver-wasm/src/engine.rs | D6 | **Parameter sweep**. Interesting approach testing multiple configs. Uses println for debugging (not ideal). Tests acceptance mode behavior. |
| 19 | ruvbot/tests/unit/security/aidefence-guard.test.ts | D9 | **Excellent behavioral**. Tests prompt injection, jailbreak, PII detection with realistic inputs. Good describe/it structure. Uses beforeEach for isolation. |
| 20 | agentic-synth/tests/unit/generators/data-generator.test.js | D7 | **Well-structured**. Tests constructor defaults, custom options, generate count, error cases. Good assertion specificity. |

### 4.2 Quality Patterns Summary

| Quality Metric | Finding |
|----------------|---------|
| **Behavior vs Implementation** | ~85% behavioral testing (good). Most tests verify public API contracts rather than internal state. |
| **Edge Cases** | ~40% of sampled tests include edge cases. Numerical edge cases (NaN, Inf, overflow) are notably absent in D1/D5. |
| **Assertion Messages** | Only **5.3%** of assertions include custom messages (2,037 out of 38,968 total assertions). This makes failure diagnosis harder. |
| **Test Isolation** | Generally good. 9 files use `static mut` in test contexts, 34 files use lazy_static/OnceCell. No pervasive shared-state issues. |
| **Flaky Indicators** | **67 test files** use sleep/thread::sleep. **72 files** use rand in test code. **287 files** reference SystemTime/Instant. **45 tests are #[ignore]d**. These represent potential flakiness vectors. |
| **TODO/FIXME in Tests** | **102 occurrences** of TODO/FIXME/HACK/XXX in test-related code, indicating known incompleteness. |

---

## 5. Critical Untested Areas (Consolidated from Phase 2)

### D1: Vector Core
- **NaN/Inf input to distance functions**: Zero tests found for NaN propagation in cosine, euclidean, dot product calculations
- **HNSW delete under concurrent load**: Basic delete tests exist but no concurrent delete+insert stress tests
- **micro-hnsw-wasm**: Entirely untested (1,262 LOC WASM binding layer)

### D2: Storage & Persistence
- **Transaction ACID properties**: No transaction isolation, atomicity, or rollback tests found
- **Concurrent mutations**: No concurrent write tests for collection operations
- **Parser recursion bombs**: No fuzz-style tests for Cypher/SPARQL parsers
- **ruvector-collections**: Only 4 tests for the entire collection management layer

### D3: Distributed Systems (HIGHEST RISK)
- **Everything multi-node**: Zero integration tests across all 4 crates
- **Network partitions**: Zero partition tolerance tests
- **Raft leader election**: Only single-node log tests; no election, term conflict, or split-brain tests
- **Snapshot transfer**: ruvector-snapshot has 5 basic tests; no cross-node transfer tests
- **Replication failover**: ruvector-replication has 23 unit tests but no failover simulation

### D4: API & Server
- **ruvector-server**: Zero tests for the primary HTTP API (health, readiness, points, collections routes)
- **SQL injection paths**: Only 1 injection test found (in tenancy validation)
- **Page boundary handling**: No tests for pagination edge cases

### D5: ML & Neural
- **Adversarial inputs**: No tests for adversarial/malformed tensor inputs
- **Numerical edge cases**: No NaN/Inf/subnormal tests in FlashAttention, CNN, or GNN forward passes
- **prime-radiant**: 1,606 pub fns but only 713 tests -- ~56% coverage by function count

### D6: Specialized Crates
- **rvf-cli** (2,073 LOC), **rvf-server** (2,363 LOC), **rvf-node** (1,058 LOC): Zero tests
- **sona**: 87 tests for 402 pub fns (22% coverage by function count)
- **ruvector-fpga-transformer**: Only 64 tests for 7,582 LOC
- **neural-trader-***: 4 crates with minimal tests (2, 7, 3, 10 respectively)

### D7: WASM & Node Bindings
- **16 WASM crates with zero tests** (10,270 LOC): No verification that WASM bindings correctly wrap underlying Rust functions
- **8 NAPI-RS Node crates with zero tests** (6,020 LOC): Memory safety at FFI boundary completely untested
- **18 NPM packages with zero JS tests** (63,839 LOC): npm/ruvector alone is 28,650 LOC with zero tests

### D8: LLM & Inference
- **ruvllm-cli**: Only 11 tests for CLI integration; no end-to-end inference pipeline tests
- **Metal/GPU tests**: Silently skip on non-Metal hardware -- coverage is platform-dependent
- **Zero benchmarks**: No performance regression detection for the inference engine

### D9: UI
- **83/84 Svelte components untested**: Only MarkdownRenderer has a component test
- **13 route pages**: Zero route-level tests
- **No e2e browser tests**: No Playwright/Cypress test files found

### D10: MCP & Brain
- **mcp-brain-server**: 146 tests for 357 pub fns (41% by function count)
- **agentic-robotics-mcp**: Zero tests (506 LOC)
- **agentic-robotics-node**: Zero tests (233 LOC)
- **No integration tests**: Zero cross-service integration test files

---

## 6. Recommended Test Additions

### P0: Critical (Would catch Phase 2 CRITICAL findings)

| ID | Domain | Target | Test Type | Risk Justification | Est. Tests |
|----|--------|--------|-----------|---------------------|------------|
| P0-1 | D3 | ruvector-raft | Integration | Raft leader election, term conflicts, log replication across 3+ simulated nodes | 15-20 |
| P0-2 | D3 | ruvector-replication | Integration | Failover simulation: primary failure, replica promotion, data consistency check | 10-15 |
| P0-3 | D3 | ruvector-cluster | Integration | Cluster join/leave, node discovery, partition healing | 10-15 |
| P0-4 | D4 | ruvector-server | Unit+Integration | HTTP API endpoint tests: health, readiness, CRUD for points and collections | 20-25 |
| P0-5 | D2 | ruvector-postgres | Integration | Transaction ACID: begin/commit/rollback, isolation levels, concurrent write conflicts | 15-20 |
| P0-6 | D1 | ruvector-core | Unit | NaN/Inf/subnormal inputs to all distance functions (cosine, euclidean, dot, manhattan) | 12-16 |
| P0-7 | D1 | ruvector-core | Stress | Concurrent HNSW insert+delete+search under load (100+ threads) | 5-8 |
| P0-8 | D2 | ruvector-postgres | Security | SQL injection tests for Cypher, SPARQL, and SQL query paths | 10-15 |

**P0 Total: ~97-134 tests**

### P1: High (Would catch Phase 2 HIGH findings)

| ID | Domain | Target | Test Type | Risk Justification | Est. Tests |
|----|--------|--------|-----------|---------------------|------------|
| P1-1 | D5 | prime-radiant, ruvector-attention | Unit | NaN/Inf/zero-length tensor inputs to forward passes | 20-25 |
| P1-2 | D5 | ruvector-cnn | Unit | Adversarial input dimensions, zero-size kernels, overflow conditions | 10-15 |
| P1-3 | D7 | ruvector-node, ruvector-attention-node | Unit | NAPI binding smoke tests for all exported functions | 20-30 |
| P1-4 | D7 | npm/ruvector (28K LOC) | Unit | Core API surface tests for the primary JS client | 30-40 |
| P1-5 | D8 | ruvllm-cli | Integration | End-to-end CLI inference with model loading and generation | 8-12 |
| P1-6 | D2 | ruvector-postgres | Fuzz | Cypher/SPARQL parser recursion bomb and malformed input tests | 10-15 |
| P1-7 | D6 | sona | Unit | Coverage for high-usage pub fns (402 pub fns, only 87 tests) | 20-30 |
| P1-8 | D6 | rvf-cli, rvf-server | Unit+Integration | CLI command tests, server endpoint tests | 15-20 |
| P1-9 | D10 | mcp-brain-server | Integration | Cross-module integration: store -> embed -> search -> retrieve pipeline | 10-15 |
| P1-10 | D3 | ruvector-delta-consensus | Unit | Byzantine fault scenarios, conflicting delta application | 8-12 |

**P1 Total: ~151-214 tests**

### P2: General Coverage Improvements

| ID | Domain | Target | Test Type | Justification | Est. Tests |
|----|--------|--------|-----------|---------------|------------|
| P2-1 | D7 | 16 WASM crates | Smoke | Basic WASM instantiation and function call for each crate | 32-48 |
| P2-2 | D9 | Svelte components | Component | Top 20 most-used components need render tests | 20-40 |
| P2-3 | D9 | Route pages | E2E | Playwright smoke tests for all 13 routes | 13-26 |
| P2-4 | D6 | neural-trader-* | Unit | Basic functionality tests for all 4 crates | 15-20 |
| P2-5 | D1 | ruvector-math | Unit | Increase coverage for 403 pub fns (currently 148 tests) | 30-50 |
| P2-6 | D2 | ruvector-collections | Unit | Expand from 4 tests to cover all 21 pub fns | 15-20 |
| P2-7 | D4 | ruvector-router-cli, ruvector-router-ffi | Unit | Basic tests for CLI args and FFI boundary | 10-15 |
| P2-8 | D8 | ruvllm | Benchmark | Performance regression benchmarks for key inference paths | 10-15 |
| P2-9 | D10 | agentic-robotics-mcp, -node | Unit | Basic functionality tests | 8-12 |
| P2-10 | All | Assertion messages | Refactor | Add custom messages to top ~500 bare assertions in critical paths | N/A |
| P2-11 | D7 | npm/postgres-cli (10K LOC) | Unit | CLI command tests for database management | 15-20 |
| P2-12 | D6 | ruvector-fpga-transformer | Unit | Expand from 64 tests for 7,582 LOC | 20-30 |

**P2 Total: ~188-296 tests**

---

## 7. Risk-Weighted Prioritization Matrix

```
Risk Score = (Change Impact * Complexity * Criticality) / Test Coverage

  Highest Risk (Score > 8):
  +-------------------------------------------------------+
  | D3: Distributed Systems (entire domain)        [9.5]  |
  |   - Zero multi-node tests, production-critical         |
  |   - Raft consensus bugs = data loss                    |
  +-------------------------------------------------------+
  | D4: ruvector-server (zero tests)               [9.2]  |
  |   - Primary API entry point, security boundary         |
  +-------------------------------------------------------+
  | D2: Transaction/ACID gaps                      [8.8]  |
  |   - 65K LOC storage layer, data integrity critical     |
  +-------------------------------------------------------+

  High Risk (Score 6-8):
  +-------------------------------------------------------+
  | D1: NaN distance functions                     [7.5]  |
  | D5: Adversarial tensor inputs                  [7.2]  |
  | D7: npm/ruvector (28K LOC, 0 tests)            [7.0]  |
  | D7: NAPI-RS memory safety boundary             [6.8]  |
  | D2: SQL injection paths                        [6.5]  |
  +-------------------------------------------------------+

  Medium Risk (Score 4-6):
  +-------------------------------------------------------+
  | D8: ruvllm-cli integration                     [5.5]  |
  | D6: sona, rvf-cli, rvf-server                  [5.2]  |
  | D10: mcp-brain-server integration              [5.0]  |
  | D9: Svelte component coverage                  [4.5]  |
  | D7: WASM binding smoke tests                   [4.2]  |
  +-------------------------------------------------------+
```

---

## 8. Summary Statistics

| Metric | Value |
|--------|-------|
| Total Rust crates | 113 |
| Total Rust production LOC | 896,197 |
| Total #[test] functions | 15,857 |
| Files with inline test modules | 1,632 |
| Dedicated test files (tests/) | 265 |
| Benchmark files | 150 |
| JS/TS test files | 81 |
| Crates with zero tests | 31 (16,290 LOC) |
| WASM crates with zero tests | 16 (10,270 LOC) |
| NAPI-RS crates with zero tests | 8 (6,020 LOC) |
| NPM packages with zero tests | 18 (63,839 LOC) |
| Total assertions | 38,968 |
| Assertions with custom messages | 2,037 (5.3%) |
| #[ignore]d tests | 45 |
| Files with sleep in test code | 67 |
| TODO/FIXME in test code | 102 |
| Overall test:code LOC ratio | 1:6.9 |
| Overall test fn:pub fn ratio | ~0.50 |
| **P0 recommended test additions** | **97-134** |
| **P1 recommended test additions** | **151-214** |
| **P2 recommended test additions** | **188-296** |
| **Total recommended additions** | **436-644** |

---

## 9. Corrections to Phase 2 Findings

| Phase 2 Claim | Actual Finding |
|----------------|----------------|
| rvagent-wasm has 0 tests (D6) | rvagent-wasm has **61 tests** across 8 inline modules (backends, bridge, gallery, tools, lib, mcp, rvf) |
| 10 crates with zero tests (D6) | 31 crates with zero tests repo-wide; many D6 crates are well-tested |
| 8/10 NAPI packages with zero JS tests (D7) | Confirmed: 8 NAPI-RS Rust crates have zero tests; additionally 18 NPM packages have zero JS tests |

---

*Generated by QE Coverage Specialist (V3) -- Phase 3 Cross-Cutting Analysis*
