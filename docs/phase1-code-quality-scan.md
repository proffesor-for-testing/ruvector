# Phase 1: Code Quality Scan Results

**Date**: 2026-03-29
**Branch**: qe-working-branch
**Scope**: All Rust source files (*.rs) under crates/
**Total Files Scanned**: 2,535
**Total Lines of Code**: 1,100,520

---

## Step 1.4: File Size Violations (>500 LOC)

### Summary

| Metric | Count |
|--------|-------|
| Total files > 500 LOC | **826** (32.6% of all .rs files) |
| Files > 2000 LOC | **16** (CRITICAL) |
| Files > 1000 LOC | **169** (HIGH) |
| Files 501-1000 LOC | **657** (MEDIUM) |
| Files <= 500 LOC | 1,709 (67.4% compliant) |

### Domain Breakdown

| Domain | ID | Files >500 LOC | Total LOC in Violations | Severity |
|--------|----|----------------|------------------------|----------|
| Specialized/Research | D10 | 625 | 526,769 | CRITICAL (volume) |
| Security & Persistence | D4 | 57 | 49,038 | HIGH |
| Neural/ML | D5 | 52 | 35,542 | HIGH |
| WASM Bindings | D6 | 27 | 21,302 | MEDIUM |
| Core Vector DB | D1 | 24 | 16,908 | MEDIUM |
| Graph Database | D2 | 21 | 19,066 | MEDIUM |
| Distributed Systems | D3 | 9 | 5,567 | LOW |
| CLI & Router | D8 | 6 | 5,929 | LOW |
| Node.js Bindings | D7 | 5 | 4,657 | LOW |
| UI Layer | D9 | 0 | 0 | PASS |

### CRITICAL: Files Exceeding 2000 LOC (16 files)

| # | File | LOC | Domain | Crate |
|---|------|-----|--------|-------|
| 1 | crates/mcp-brain-server/src/routes.rs | 6,807 | D10 | mcp-brain-server |
| 2 | crates/ruvllm/src/bitnet/backend.rs | 4,843 | D10 | ruvllm |
| 3 | crates/rvf/rvf-runtime/src/store.rs | 2,766 | D10 | rvf |
| 4 | crates/ruvector-cli/src/cli/hooks.rs | 2,507 | D8 | ruvector-cli |
| 5 | crates/rvlite/src/sparql/parser.rs | 2,502 | D10 | rvlite |
| 6 | crates/ruvector-postgres/src/graph/sparql/parser.rs | 2,496 | D4 | ruvector-postgres |
| 7 | crates/ruvector-postgres/src/index/hnsw_am.rs | 2,351 | D4 | ruvector-postgres |
| 8 | crates/ruvector-temporal-tensor/src/store.rs | 2,283 | D10 | ruvector-temporal-tensor |
| 9 | crates/ruvllm/src/kernels/attention.rs | 2,214 | D10 | ruvllm |
| 10 | crates/ruvector-postgres/src/index/ivfflat_am.rs | 2,174 | D4 | ruvector-postgres |
| 11 | crates/ruvllm/src/backends/coreml_backend.rs | 2,169 | D10 | ruvllm |
| 12 | crates/ruvllm/src/training/tool_dataset.rs | 2,146 | D10 | ruvllm |
| 13 | crates/ruvector-postgres/src/distance/simd.rs | 2,128 | D4 | ruvector-postgres |
| 14 | crates/ruQu/src/tile.rs | 2,124 | D10 | ruQu |
| 15 | crates/ruqu-core/src/decoder.rs | 2,103 | D10 | ruqu-core |
| 16 | crates/ruvllm/src/kernels/matmul.rs | 2,049 | D10 | ruvllm |

### P0 Critical Domains: Detailed Violations

#### D1 - Core Vector DB (24 files >500 LOC)

| File | LOC | Category |
|------|-----|----------|
| ruvector-core/src/simd_intrinsics.rs | 1,670 | src |
| ruvector-core/src/agenticdb.rs | 1,447 | src |
| ruvector-core/src/quantization.rs | 934 | src |
| ruvector-core/src/embeddings.rs | 833 | src |
| ruvector-core/tests/test_memory_pool.rs | 772 | test |
| ruvector-core/tests/test_quantization.rs | 767 | test |
| ruvector-core/src/advanced_features/sparse_vector.rs | 753 | src |
| ruvector-core/src/arena.rs | 704 | src |
| ruvector-core/src/advanced_features/graph_rag.rs | 699 | src |
| ruvector-core/src/advanced_features/matryoshka.rs | 642 | src |
| ruvector-core/src/lockfree.rs | 590 | src |
| ruvector-core/src/advanced_features/multi_vector.rs | 565 | src |
| ruvector-core/tests/unit_tests.rs | 555 | test |
| ruvector-core/tests/test_simd_correctness.rs | 552 | test |
| ruvector-core/tests/advanced_features_integration.rs | 550 | test |
| ruvector-core/src/advanced_features/product_quantization.rs | 549 | src |
| ruvector-core/src/advanced/hypergraph.rs | 545 | src |
| ruvector-core/src/advanced_features/conformal_prediction.rs | 503 | src |
| ruvector-collections/src/manager.rs | 522 | src |
| ruvector-filter/src/evaluator.rs | 593 | src |
| ruvector-math/src/product_manifold/manifold.rs | 575 | src |
| ruvector-math/src/tensor_networks/tensor_train.rs | 543 | src |
| ruvector-math/src/optimal_transport/sliced_wasserstein.rs | 533 | src |
| ruvector-math/src/optimization/polynomial.rs | 512 | src |

#### D2 - Graph Database (21 files >500 LOC)

| File | LOC | Category |
|------|-----|----------|
| ruvector-graph-transformer/src/temporal.rs | 1,855 | src |
| ruvector-graph-transformer/src/manifold.rs | 1,738 | src |
| ruvector-graph-transformer/src/biological.rs | 1,670 | src |
| ruvector-graph-transformer/src/verified_training.rs | 1,419 | src |
| ruvector-graph/src/cypher/parser.rs | 1,295 | src |
| ruvector-graph-transformer/src/proof_gated.rs | 1,156 | src |
| ruvector-graph-transformer/src/physics.rs | 1,035 | src |
| ruvector-graph-transformer/src/self_organizing.rs | 1,007 | src |
| ruvector-graph-transformer/src/economic.rs | 864 | src |
| ruvector-graph/tests/transaction_tests.rs | 818 | test |
| ruvector-graph/src/distributed/gossip.rs | 623 | src |
| ruvector-graph/src/cypher/semantic.rs | 616 | src |
| ruvector-graph/src/distributed/shard.rs | 595 | src |
| ruvector-graph/src/cypher/optimizer.rs | 582 | src |
| ruvector-graph/src/distributed/federation.rs | 582 | src |
| ruvector-graph-wasm/src/lib.rs | 569 | src |
| ruvector-graph-node/src/lib.rs | 563 | src |
| ruvector-graph/src/distributed/coordinator.rs | 535 | src |
| ruvector-graph/src/executor/operators.rs | 521 | src |
| ruvector-graph/src/distributed/rpc.rs | 515 | src |
| ruvector-graph-transformer/tests/integration.rs | 508 | test |

#### D3 - Distributed Systems (9 files >500 LOC)

| File | LOC | Category |
|------|-----|----------|
| ruvector-delta-index/src/lib.rs | 774 | src |
| ruvector-delta-core/src/delta.rs | 692 | src |
| ruvector-delta-core/src/compression.rs | 680 | src |
| ruvector-raft/src/node.rs | 631 | src |
| ruvector-delta-wasm/src/lib.rs | 604 | src |
| ruvector-delta-core/src/encoding.rs | 601 | src |
| ruvector-delta-graph/src/lib.rs | 562 | src |
| ruvector-cluster/src/lib.rs | 513 | src |
| ruvector-delta-core/src/window.rs | 510 | src |

### Top Violators by Crate (>500 LOC file count)

| Crate | Domain | Files >500 LOC |
|-------|--------|----------------|
| ruvllm | D10 | 158 |
| ruvix | D10 | 81 |
| prime-radiant | D10 | 66 |
| ruvector-postgres | D4 | 57 |
| ruvector-mincut | D10 | 57 |
| rvf | D10 | 35 |
| rvAgent | D10 | 31 |
| ruvector-mincut-gated-transformer | D10 | 25 |
| ruQu | D10 | 25 |
| ruvector-nervous-system | D10 | 23 |
| ruqu-core | D10 | 23 |
| ruvector-core | D1 | 18 |
| ruvector-cnn | D5 | 17 |
| mcp-brain-server | D10 | 15 |
| ruvector-temporal-tensor | D10 | 14 |
| ruvllm-wasm | D6 | 12 |
| rvlite | D10 | 11 |
| ruvector-solver | D10 | 10 |
| ruvector-graph | D2 | 10 |
| ruvector-gnn | D5 | 10 |
| ruvector-attention | D5 | 10 |
| ruvector-graph-transformer | D2 | 9 |
| sona | D5 | 9 |

---

## Step 1.5: unwrap() Audit

### Aggregate Summary

| Category | Files with unwrap() | Total unwrap() Calls |
|----------|--------------------|--------------------|
| Library code (src/*.rs) | **1,027** | **8,315** |
| Test code (tests/*.rs) | **204** | **4,165** |
| Bench code (benches/*.rs) | **79** | (included in totals) |
| **Grand Total** | **1,310** | **12,480+** |

### P0 Critical Domains: unwrap() in Library Code

#### D1 - Core Vector DB

| Crate | unwrap() Count | Files Affected |
|-------|---------------|----------------|
| ruvector-core | 221 | 23 |
| ruvector-filter | 62 | 4 |
| ruvector-math | 44 | 11 |
| ruvector-metrics | 14 | 2 |
| ruvector-collections | 4 | 3 |
| **D1 Total** | **345** | **43** |

D1 Worst Offenders:

| File | unwrap() Count | LOC |
|------|---------------|-----|
| ruvector-core/src/advanced_features/matryoshka.rs | 29 | 642 |
| ruvector-filter/src/lib.rs | 29 | ~500 |
| ruvector-core/src/advanced_features/multi_vector.rs | 26 | 565 |
| ruvector-core/src/advanced_features/opq.rs | 20 | ~500 |
| ruvector-core/src/advanced_features/conformal_prediction.rs | 18 | 503 |
| ruvector-core/src/embeddings.rs | 16 | 833 |
| ruvector-filter/src/evaluator.rs | 16 | 593 |
| ruvector-filter/src/index.rs | 15 | ~500 |
| ruvector-core/src/storage_memory.rs | 15 | ~500 |
| ruvector-core/src/advanced_features/diskann.rs | 15 | ~500 |

#### D2 - Graph Database

| Crate | unwrap() Count | Files Affected |
|-------|---------------|----------------|
| ruvector-graph-transformer | 96 | 9 |
| ruvector-graph | 91 | 23 |
| ruvector-graph-wasm | 11 | 2 |
| ruvector-graph-node | 4 | 1 |
| **D2 Total** | **202** | **35** |

D2 Worst Offenders:

| File | unwrap() Count | LOC |
|------|---------------|-----|
| ruvector-graph/src/executor/cache.rs | 18 | ~500 |
| ruvector-graph-transformer/src/biological.rs | 18 | 1,670 |
| ruvector-graph-transformer/src/temporal.rs | 15 | 1,855 |
| ruvector-graph-transformer/src/physics.rs | 15 | 1,035 |
| ruvector-graph/src/graph.rs | 12 | ~500 |
| ruvector-graph-transformer/src/self_organizing.rs | 12 | 1,007 |
| ruvector-graph-transformer/src/manifold.rs | 11 | 1,738 |
| ruvector-graph-transformer/src/economic.rs | 11 | 864 |
| ruvector-graph/src/executor/stats.rs | 9 | ~500 |
| ruvector-graph-transformer/src/verified_training.rs | 8 | 1,419 |

#### D3 - Distributed Systems

| Crate | unwrap() Count | Files Affected |
|-------|---------------|----------------|
| ruvector-delta-core | 29 | 6 |
| ruvector-replication | 27 | 5 |
| ruvector-delta-wasm | 24 | 3 |
| ruvector-delta-index | 19 | 2 |
| ruvector-cluster | 17 | 3 |
| ruvector-delta-consensus | 11 | 2 |
| ruvector-raft | 8 | 3 |
| ruvector-delta-graph | 7 | 3 |
| **D3 Total** | **142** | **27** |

D3 Worst Offenders:

| File | unwrap() Count | LOC |
|------|---------------|-----|
| ruvector-replication/src/replica.rs | 9 | ~500 |
| ruvector-cluster/src/consensus.rs | 7 | ~500 |
| ruvector-cluster/src/lib.rs | 6 | 513 |
| ruvector-replication/src/sync.rs | 6 | ~500 |
| ruvector-replication/src/failover.rs | 6 | ~500 |
| ruvector-replication/src/conflict.rs | 5 | ~500 |
| ruvector-raft/src/rpc.rs | 4 | ~500 |
| ruvector-cluster/src/discovery.rs | 4 | ~500 |
| ruvector-raft/src/log.rs | 3 | ~500 |

### Top 30 Worst Offenders: unwrap() in Library Code (All Domains)

| # | File | unwrap() | Domain | LOC |
|---|------|---------|--------|-----|
| 1 | ruvector-mincut/src/euler/mod.rs | 107 | D10 | 1,244 |
| 2 | rvf/rvf-runtime/src/store.rs | 104 | D10 | 2,766 |
| 3 | ruqu-core/src/clifford_t.rs | 91 | D10 | 1,004 |
| 4 | ruvector-gnn/src/training.rs | 78 | D5 | 1,367 |
| 5 | ruvector-mincut/src/witness/mod.rs | 71 | D10 | 920 |
| 6 | ruvector-temporal-tensor/src/store.rs | 64 | D10 | 2,283 |
| 7 | ruvector-mincut/src/linkcut/mod.rs | 63 | D10 | 962 |
| 8 | rvf/rvf-adapters/rvlite/src/collection.rs | 62 | D10 | ~500 |
| 9 | ruvector-postgres/src/graph/operators.rs | 62 | D4 | 1,067 |
| 10 | rvf/rvf-adapters/agentic-flow/src/swarm_store.rs | 61 | D10 | ~500 |
| 11 | ruvllm/src/quantize/turbo_quant.rs | 56 | D10 | 1,483 |
| 12 | ruvector-temporal-tensor/src/persistence.rs | 55 | D10 | 859 |
| 13 | ruvix/crates/fs/src/ramfs.rs | 52 | D10 | 945 |
| 14 | ruvix/benches/src/report.rs | 49 | D10 | ~500 |
| 15 | rvAgent/rvagent-core/src/cow_state.rs | 48 | D10 | 730 |
| 16 | rvf/rvf-adapters/claude-flow/src/memory_store.rs | 47 | D10 | ~500 |
| 17 | rvlite/src/sparql/triple_store.rs | 46 | D10 | ~500 |
| 18 | rvAgent/rvagent-backends/src/sandbox.rs | 45 | D10 | 777 |
| 19 | ruvector-mincut/src/sparsify/mod.rs | 43 | D10 | 843 |
| 20 | rvf/rvf-server/src/http.rs | 42 | D10 | 1,630 |
| 21 | rvf/rvf-adapters/ospipe/src/observation_store.rs | 42 | D10 | ~500 |
| 22 | ruvix/crates/physmem/src/frame.rs | 42 | D10 | 728 |
| 23 | ruvector-mincut/src/tree/mod.rs | 42 | D10 | 770 |
| 24 | rvAgent/rvagent-mcp/src/protocol.rs | 41 | D10 | 767 |
| 25 | ruvector-mincut/src/algorithm/mod.rs | 41 | D10 | 1,008 |
| 26 | ruvector-mincut/src/graph/mod.rs | 41 | D10 | 734 |
| 27 | rvAgent/rvagent-acp/src/server.rs | 39 | D10 | ~500 |
| 28 | ruqu-core/src/stabilizer.rs | 39 | D10 | 789 |
| 29 | ruvector-core/src/advanced_features/matryoshka.rs | 29 | D1 | 642 |
| 30 | ruvector-filter/src/lib.rs | 29 | D1 | ~500 |

### Test Code unwrap() (Top 15 -- Expected/Acceptable)

| File | unwrap() Count |
|------|---------------|
| ruqu-core/tests/test_state.rs | 216 |
| prime-radiant/tests/storage_tests.rs | 128 |
| ruvix/crates/region/tests/region_test.rs | 117 |
| ruvix/crates/fs/tests/fs_test.rs | 108 |
| rvAgent/rvagent-mcp/tests/integration.rs | 100 |
| rvf/tests/rvf-integration/tests/e2e_store_lifecycle.rs | 77 |
| ruvix/tests/tests/syscall_flows.rs | 73 |
| ruvix/crates/nucleus/tests/deterministic_replay.rs | 72 |
| rvf/tests/rvf-integration/tests/computational_container.rs | 62 |
| rvf/tests/rvf-integration/tests/cross_platform_compat.rs | 61 |
| rvf/tests/rvf-integration/tests/segment_preservation.rs | 58 |
| ruvix/tests/tests/adr087_section17_acceptance.rs | 57 |
| ruqu-algorithms/tests/test_algorithms.rs | 56 |
| ruvector-graph/tests/transaction_tests.rs | 54 |
| ruqu-exotic/tests/test_exotic.rs | 54 |

---

## Risk Matrix

### Combined Risk: File Size + unwrap() Density

Files that are both oversized AND have high unwrap() density represent the highest risk for production panics.

| Risk Level | Criteria | Count |
|------------|----------|-------|
| CRITICAL | >1000 LOC + >40 unwrap() | 8 |
| HIGH | >500 LOC + >20 unwrap() | ~25 |
| MEDIUM | >500 LOC + >10 unwrap() | ~60 |

#### CRITICAL Risk Files (>1000 LOC and >40 unwrap()):

| File | LOC | unwrap() | Domain |
|------|-----|---------|--------|
| ruvector-mincut/src/euler/mod.rs | 1,244 | 107 | D10 |
| rvf/rvf-runtime/src/store.rs | 2,766 | 104 | D10 |
| ruqu-core/src/clifford_t.rs | 1,004 | 91 | D10 |
| ruvector-gnn/src/training.rs | 1,367 | 78 | D5 |
| ruvector-mincut/src/linkcut/mod.rs | 962 | 63 | D10 |
| ruvector-temporal-tensor/src/store.rs | 2,283 | 64 | D10 |
| ruvector-mincut/src/witness/mod.rs | 920 | 71 | D10 |
| ruvllm/src/quantize/turbo_quant.rs | 1,483 | 56 | D10 |

---

## Recommendations

### Immediate Actions (P0)

1. **D1 ruvector-core/src/simd_intrinsics.rs (1,670 LOC)**: Split into per-architecture modules (avx2.rs, neon.rs, generic.rs). This is critical infrastructure code.

2. **D1 ruvector-core/src/agenticdb.rs (1,447 LOC)**: Extract query builder, connection pool, and schema management into separate modules.

3. **D2 ruvector-graph-transformer/src/temporal.rs (1,855 LOC)**: Decompose into temporal_attention.rs, temporal_encoding.rs, temporal_aggregation.rs.

4. **D3 unwrap() cleanup**: Replace all 142 unwrap() calls in D3 crates with proper error handling. Distributed systems code must never panic on recoverable errors.

### Short-Term Actions (P1)

5. **All >2000 LOC files**: Each of the 16 files exceeding 2,000 lines needs a decomposition plan. Priority on D4 (ruvector-postgres) which has 4 files in this bracket.

6. **unwrap() in D1 core**: Reduce the 345 unwrap() calls to <50 using `?` operator and proper `Result`/`Option` handling. Focus on matryoshka.rs (29), multi_vector.rs (26), and opq.rs (20) first.

7. **D2 Graph Transformer**: 8 of 9 source files exceed 500 LOC. The entire crate needs a structural review.

### Long-Term Actions (P2)

8. **Crate-level decomposition**: ruvllm (158 files >500 LOC) and ruvix (81 files >500 LOC) are structural outliers. Consider splitting into sub-crates.

9. **Enforce CI gate**: Add a CI check that fails on new files exceeding 500 LOC and new unwrap() in D1-D3 library code.

10. **unwrap() budget**: Establish per-domain unwrap() budgets. Target: 0 unwrap() in D3 (distributed), <10 in D1 (core), <20 in D2 (graph) library code.

---

## Appendix: Domain Mapping Reference

| Domain | ID | Crates |
|--------|----|--------|
| Core Vector DB | D1 | ruvector-core, ruvector-collections, ruvector-filter, ruvector-math, ruvector-metrics |
| Graph Database | D2 | ruvector-graph, ruvector-graph-node, ruvector-graph-wasm, ruvector-graph-transformer |
| Distributed Systems | D3 | ruvector-raft, ruvector-replication, ruvector-cluster, ruvector-delta-* |
| Security & Persistence | D4 | ruvector-postgres, ruvector-server, ruvector-snapshot, ruvector-verified |
| Neural/ML | D5 | ruvector-attention, ruvector-cnn, ruvector-gnn, neural-trader-*, sona |
| WASM Bindings | D6 | All *-wasm crates |
| Node.js Bindings | D7 | All *-node crates |
| CLI & Router | D8 | ruvector-cli, ruvector-router-*, ruvllm-cli |
| UI Layer | D9 | ui/ruvocal |
| Specialized/Research | D10 | Everything else (ruvllm, ruvix, prime-radiant, rvf, rvAgent, ruQu, mcp-brain-server, etc.) |
