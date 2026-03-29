# Phase 1 - Step 1.9: Dependency Tree Analysis

**Date**: 2026-03-29
**Scope**: Full RuVector monorepo (Rust workspace + NPM packages)
**Analyzer**: V3 QE Dependency Mapper

---

## 1. Workspace Structure

The root `Cargo.toml` defines a Rust workspace with **resolver v2** and the following composition:

| Category | Count | Notes |
|----------|-------|-------|
| Workspace members (declared) | ~120 entries | Includes crates/, examples/, and nested sub-workspaces |
| Workspace excludes | 14 entries | micro-hnsw-wasm, hyperbolic-hnsw variants, rvf/*, mcp-brain-server, edge examples |
| Total parsed crates | **165** | All crates/ + ruvix/ + rvAgent/ + examples/ |
| Workspace-level deps | 36 | Defined in `[workspace.dependencies]` for consistency |

### Workspace Package Defaults

- **Version**: 2.1.0
- **Edition**: 2021
- **Rust version**: 1.77
- **License**: MIT

### Sub-workspaces (Excluded, Self-contained)

The `crates/rvf/` directory is an excluded sub-workspace containing **18 sub-crates** (rvf-types, rvf-crypto, rvf-wire, rvf-runtime, rvf-kernel, rvf-ebpf, rvf-quant, rvf-federation, rvf-index, rvf-manifest, rvf-launch, rvf-server, rvf-import, rvf-adapters, rvf-node, rvf-wasm, rvf-solver-wasm, rvf-cli). These are referenced by path from several workspace members (mcp-brain-server, benchmarks, domain-expansion, robotics, rvlite, rvf-examples, rvf-kernel-optimized).

---

## 2. Internal Dependency Map

### 2.1 Total Internal Dependency Edges: **283**

Average internal dependencies per crate: **1.7**

### 2.2 Most Depended-Upon Crates (Highest Fan-In)

These are the foundational crates that many others build upon. Changes here have the widest blast radius.

| Rank | Crate | Fan-In (Ca) | Fan-Out (Ce) | Instability (I) | Risk |
|------|-------|-------------|--------------|------------------|------|
| 1 | **ruvector-core** | **29** | 0 | 0.00 | Low |
| 2 | **ruvix-types** | **23** | 0 | 0.00 | Low |
| 3 | ruvector-gnn | 12 | 1 | 0.08 | Low |
| 4 | ruvector-attention | 12 | 1 | 0.08 | Low |
| 5 | ruvector-mincut | 11 | 2 | 0.15 | Low |
| 6 | ruvector-graph | 9 | 4 | 0.31 | Low |
| 7 | ruvector-solver | 8 | 0 | 0.00 | Low |
| 8 | rvf-types | 7 | 0 | 0.00 | Low |
| 9 | ruvix-cap | 7 | 1 | 0.12 | Low |
| 10 | ruvix-region | 7 | 1 | 0.12 | Low |
| 11 | rvagent-core | 7 | 0 | 0.00 | Low |
| 12 | ruvector-delta-core | 6 | 0 | 0.00 | Low |
| 13 | agentic-robotics-core | 5 | 0 | 0.00 | Low |
| 14 | rvf-runtime | 5 | 0 | 0.00 | Low |
| 15 | ruvix-queue | 5 | 2 | 0.29 | Low |

**Key finding**: `ruvector-core` is the single most critical crate (29 dependents, zero outgoing deps). It is a pure foundation -- any breaking change here propagates to ~18% of the entire workspace.

`ruvix-types` plays an identical role within the RuVix Cognition Kernel sub-ecosystem (23 dependents, zero outgoing deps).

### 2.3 Crates with Most Dependencies (Highest Fan-Out)

These are integration/aggregation crates that pull in the most internal dependencies.

| Rank | Crate | Fan-Out (Ce) | Fan-In (Ca) | Instability (I) | Role |
|------|-------|--------------|-------------|------------------|------|
| 1 | **mcp-brain-server** | **13** | 0 | 1.00 | Integration hub |
| 2 | rvf-examples | 13 | 0 | 1.00 | Example aggregator |
| 3 | ospipe | 10 | 0 | 1.00 | Pipeline integration |
| 4 | ruvix-demo | 9 | 0 | 1.00 | Demo aggregator |
| 5 | rvdna | 9 | 0 | 1.00 | DNA analysis tool |
| 6 | prime-radiant | 7 | 0 | 1.00 | Cross-domain integration |
| 7 | ruvector-postgres | 7 | 0 | 1.00 | Persistence integration |
| 8 | ruvector-graph-transformer | 6 | 0 | 1.00 | Graph ML bridge |
| 9 | ruvllm | 6 | 2 | 0.75 | LLM integration |
| 10 | rvf-kernel-optimized | 6 | 0 | 1.00 | Optimized kernel example |

**Key finding**: `mcp-brain-server` has the highest fan-out (13 internal deps) -- it aggregates from distributed, neural/ML, specialized, and RVF domains. This is appropriate for a server integration crate but it represents a deployment coupling risk. Similarly, `ruvector-postgres` pulls from 7 internal crates across 4 different domains.

### 2.4 Circular Dependencies

**Result: NONE DETECTED**

No circular dependency cycles were found across all 165 crates and 283 internal edges. The dependency graph is a proper DAG (directed acyclic graph).

This is a strong positive indicator of clean architectural layering.

---

## 3. External Dependency Analysis

### 3.1 Summary

| Metric | Value |
|--------|-------|
| Total unique external crate dependencies | **232** |
| Workspace-level shared dependencies | 36 |
| Most widely used dependency | serde (124 crates, 75%) |

### 3.2 Top 30 External Dependencies by Usage

| Rank | Dependency | Crate Count | Category |
|------|-----------|-------------|----------|
| 1 | serde | 124 | Serialization |
| 2 | serde_json | 107 | Serialization |
| 3 | thiserror | 81 | Error handling |
| 4 | criterion | 69 | Benchmarking |
| 5 | rand | 66 | Random generation |
| 6 | proptest | 54 | Property testing |
| 7 | tracing | 52 | Observability |
| 8 | anyhow | 50 | Error handling |
| 9 | tokio | 49 | Async runtime |
| 10 | parking_lot | 42 | Concurrency |
| 11 | wasm-bindgen | 39 | WASM bindings |
| 12 | chrono | 38 | Date/time |
| 13 | js-sys | 37 | JS interop |
| 14 | uuid | 36 | Identifiers |
| 15 | rayon | 33 | Parallelism |
| 16 | wasm-bindgen-test | 32 | WASM testing |
| 17 | dashmap | 29 | Concurrent maps |
| 18 | serde-wasm-bindgen | 29 | WASM serialization |
| 19 | getrandom | 29 | RNG seeding |
| 20 | tracing-subscriber | 28 | Observability |
| 21 | console_error_panic_hook | 26 | WASM debugging |
| 22 | tempfile | 26 | Testing utility |
| 23 | bincode | 22 | Binary serialization |
| 24 | ndarray | 20 | N-dimensional arrays |
| 25 | rand_distr | 20 | Random distributions |
| 26 | web-sys | 19 | Web API bindings |
| 27 | sha2 | 18 | Cryptographic hashing |
| 28 | approx | 16 | Floating-point comparison |
| 29 | async-trait | 15 | Async trait support |
| 30 | futures | 15 | Async primitives |

### 3.3 Potentially Duplicated Dependencies

These are cases where multiple crates serve a similar purpose. Some are legitimate (different use cases), others may warrant consolidation.

| Category | Crates Found | Assessment |
|----------|-------------|------------|
| **Serialization** | serde, serde_json, bincode, rkyv, rmp-serde | **Acceptable** -- serde is the standard; bincode/rkyv for performance; rmp for MessagePack |
| **Error handling** | thiserror, anyhow | **Acceptable** -- thiserror for library errors, anyhow for application errors (correct pattern) |
| **Random** | rand, fastrand, getrandom, rand_chacha, rand_distr | **Review** -- fastrand may be redundant if rand is available; getrandom is for seeding |
| **Hashing/Crypto** | sha2, sha3, blake3, ed25519-dalek, x25519-dalek | **Acceptable** -- different algorithms for different purposes |
| **Matrix/Linear Algebra** | nalgebra, ndarray | **Review** -- both are used; nalgebra for geometric algebra, ndarray for tensor ops. Consider consolidating if overlap exists |
| **Logging** | tracing, env_logger, tracing-subscriber | **Acceptable** -- tracing is primary; env_logger may be legacy |
| **HTTP** | reqwest, hyper | **Acceptable** -- reqwest is high-level client; hyper is low-level (reqwest depends on hyper) |
| **Testing** | proptest, quickcheck, mockall, criterion | **Review** -- both proptest and quickcheck serve similar purposes; consider standardizing on one |

### 3.4 Workspace Dependency Centralization

The workspace uses `[workspace.dependencies]` to centralize 36 key dependencies, ensuring version consistency. This is a best practice. However, 196 additional dependencies are declared locally in individual crates, which could lead to version drift.

**Recommendation**: Consider expanding `[workspace.dependencies]` to cover the top 50 most-used dependencies (at minimum: `dashmap`, `sha2`, `axum`, `ndarray`, `async-trait`, `futures`).

---

## 4. Domain Coupling Analysis

### 4.1 Domain Definitions

| Domain | ID | Crate Count | Description |
|--------|----|-------------|-------------|
| Core Vector DB | D1 | 5 | Foundation: core, collections, filter, math, metrics |
| Graph Database | D2 | 6 | Graph engine + transformer + bindings |
| Distributed Systems | D3 | 8 | Raft, replication, cluster, delta-* |
| Security & Persistence | D4 | 5 | Postgres, server, snapshot, verified |
| Neural/ML | D5 | 16 | Attention, CNN, GNN, neural-trader, sona |
| CLI & Router | D8 | 6 | CLI tools, router variants |
| Specialized | D10 | 39 | Mincut, solver, sparsifier, MCP, etc. |
| RuVix Kernel | D11 | 25 | Cognition kernel (types, region, queue, etc.) |
| rvAgent | D12 | 9 | AI agent framework |
| RuVLLM | D13 | 3 | LLM integration |

### 4.2 Domain-Level Coupling Metrics

| Domain | Ca (fan-in) | Ce (fan-out) | I (instability) | Risk |
|--------|-------------|--------------|------------------|------|
| **D1 Core VectorDB** | **7** | **0** | **0.00** | **Low** |
| D5 Neural/ML | 6 | 2 | 0.25 | Low |
| D3 Distributed | 2 | 1 | 0.33 | Low |
| D10 Specialized | 4 | 5 | 0.56 | Medium |
| D2 Graph | 3 | 5 | 0.62 | Medium |
| D4 Security & Persistence | 1 | 3 | 0.75 | **HIGH** |
| D13 RuVLLM | 1 | 3 | 0.75 | **HIGH** |
| D8 CLI | 0 | 3 | 1.00 | **HIGH** |
| D11 RuVix | 0 | 1 | 1.00 | **HIGH** |
| D12 rvAgent | 0 | 1 | 1.00 | **HIGH** |

**Interpretation**: Instability = Ce/(Ca+Ce). A value near 0 means "stable foundation" (many depend on it, it depends on few). A value near 1 means "unstable consumer" (it depends on many, few depend on it). HIGH instability for leaf domains (CLI, RuVix, rvAgent) is expected and acceptable -- they are application-layer consumers. HIGH instability for D4 (Security & Persistence) is a concern because it is a service-layer domain that should be more stable.

### 4.3 Cross-Domain Dependency Edges (46 total)

The 46 cross-domain edges break down by pattern:

| Pattern | Count | Examples | Assessment |
|---------|-------|----------|------------|
| **D* -> D1 Core** | 18 | gnn->core, graph->core, cluster->core, etc. | **Expected**: Core is the foundation |
| **D2 Graph -> D3 Distributed** | 3 | graph->cluster, graph->raft, graph->replication | **Expected**: Graph needs distributed primitives |
| **D2 Graph -> D5 Neural** | 2 | graph-transformer->attention, graph-transformer->gnn | **Expected**: Graph-transformer bridges these |
| **D4 Persist -> D5 Neural** | 2 | postgres->attention, postgres->sona | **Review**: Why does persistence need neural? |
| **D4 Persist -> D10 Special** | 3 | postgres->domain-expansion, postgres->solver, postgres->mincut-gated | **CONCERN**: Postgres is pulling specialized algorithms |
| **D10 Special -> D5 Neural** | 4 | prime-radiant->attention, prime-radiant->gnn, mcp-brain->sona, mcp-brain-server->sona | **Expected**: Integration crates bridge domains |
| **D10 Special -> D3 Distributed** | 1 | mcp-brain-server->delta-core | **Acceptable** |
| **D10 Special -> D2 Graph** | 2 | prime-radiant->graph, mincut->graph | **Expected** |
| **D11 RuVix -> D10 Special** | 1 | ruvix-sched->ruvector-coherence | **Review**: Cross-subsystem coupling |
| **D8 CLI -> D5 Neural** | 1 | cli->gnn | **Acceptable**: CLI exposes neural features |

### 4.4 Notable Coupling Concerns

**1. ruvector-postgres is a coupling magnet (7 internal deps, 4 domains)**

`ruvector-postgres` depends on: ruvector-core (D1), ruvector-attention (D5), ruvector-math (D1), ruvector-domain-expansion (D10), ruvector-mincut-gated-transformer (D10), ruvector-solver (D10), ruvector-sona (D5).

This makes the Postgres persistence layer a "God crate" that knows about attention mechanisms, solvers, and gated transformers. Consider introducing adapter interfaces so postgres only depends on abstract storage traits defined in D1 Core.

**2. ruvector-graph-transformer bridges 5 domains**

Depends on: ruvector-attention (D5), ruvector-coherence (D10), ruvector-gnn (D5), ruvector-mincut (D10), ruvector-solver (D10), ruvector-verified (D4).

This is architecturally justified as a transformer that combines graph, attention, and verification. However, it is a high-risk change target -- any modification could break downstream consumers.

**3. prime-radiant spans 4 domains**

Depends on: ruvector-core (D1), ruvector-attention (D5), ruvector-gnn (D5), ruvector-graph (D2), ruvector-mincut (D10), ruvector-nervous-system (D10), ruvector-raft (D3).

As an integration/orchestration crate, this is expected but should be monitored.

---

## 5. NPM Dependency Analysis

### 5.1 Summary

| Metric | Value |
|--------|-------|
| Total NPM packages | **55** |
| Total unique external production deps | **40** |
| Total unique external dev deps | **29** |
| Internal (@ruvector/*) dependency edges | **12** |

### 5.2 Packages with Most Dependencies

| Package | Prod Deps | Dev Deps | Role |
|---------|-----------|----------|------|
| @ruvector/agentic-integration | 16 | 13 | Distributed agent coordination |
| ruvbot | 11 | 7 | Bot interface |
| ruvector | 8 | 2 | Main Node.js package |
| @ruvector/burst-scaling | 6 | 11 | Auto-scaling |
| @ruvector/postgres-cli | 6 | 4 | Postgres CLI |
| @ruvector/agentic-synth | 5 | 9 | Synthetic data |
| @ruvector/graph-data-generator | 5 | 9 | Graph data gen |
| @ruvector/wasm | 5 | 1 | WASM unified |

### 5.3 Most Common NPM Dependencies

| Dependency | Packages | Category |
|-----------|----------|----------|
| commander | 8 | CLI |
| zod | 6 | Validation |
| chalk | 5 | Terminal styling |
| ora | 5 | Spinners |
| dotenv | 4 | Environment |
| express | 3 | HTTP server |
| @modelcontextprotocol/sdk | 3 | MCP integration |
| eventemitter3 | 3 | Events |

### 5.4 Internal NPM Dependency Map

| Package | Depended On By |
|---------|---------------|
| @ruvector/agentic-synth | @ruvector/agentic-synth-examples, @ruvector/graph-data-generator |
| @ruvector/core | @ruvector/node, ruvector |
| @ruvector/gnn | @ruvector/node, ruvector |
| @ruvector/attention | ruvector |
| @ruvector/sona | ruvector |
| @ruvector/learning-wasm | @ruvector/wasm |
| @ruvector/economy-wasm | @ruvector/wasm |
| @ruvector/exotic-wasm | @ruvector/wasm |
| @ruvector/nervous-system-wasm | @ruvector/wasm |
| @ruvector/attention-unified-wasm | @ruvector/wasm |
| @ruvector/rvf-node | @ruvector/rvf |
| @ruvector/rvf | @ruvector/rvf-mcp-server |

### 5.5 NPM Duplicate Dependencies

| Category | Libraries Found | Packages | Assessment |
|----------|----------------|----------|------------|
| **Logging** | winston + pino | @ruvector/agentic-integration uses both | **Issue**: Pick one. pino is faster; winston is more flexible |
| **Testing** | jest + vitest | jest in 2 pkgs, vitest in 6 pkgs | **Standardize on vitest** (already the majority) |
| **Web Framework** | express + fastify | express in 3 pkgs, fastify in 1 pkg | **Review**: @ruvector/agentic-integration uses both simultaneously |

### 5.6 Dev Dependency Standardization

| Dependency | Packages | Notes |
|-----------|----------|-------|
| typescript | 25 (45%) | Good coverage |
| @types/node | 24 (44%) | Consistent |
| @napi-rs/cli | 7 | Native Node.js bindings |
| vitest | 6 | Preferred test runner |
| eslint | 7 | Linting |
| tsup | 5 | Build tool |

---

## 6. Risk Summary and Recommendations

### 6.1 High-Risk Findings

| ID | Finding | Severity | Impact |
|----|---------|----------|--------|
| **R1** | `ruvector-core` has 29 dependents | HIGH | Any breaking API change cascades to 18% of workspace |
| **R2** | `ruvector-postgres` couples 4 domains (7 internal deps) | HIGH | Persistence layer is not properly abstracted |
| **R3** | 196 external deps declared locally (not via workspace) | MEDIUM | Version drift risk across 165 crates |
| **R4** | `mcp-brain-server` aggregates 13 internal deps | MEDIUM | Deployment coupling; all 13 must build together |
| **R5** | NPM: winston + pino in same package | LOW | Bloated bundle; pick one logging library |
| **R6** | NPM: jest + vitest split | LOW | Maintenance burden; standardize on vitest |

### 6.2 Positive Findings

| Finding | Assessment |
|---------|------------|
| **No circular dependencies** | Excellent. 165 crates, 283 edges, zero cycles |
| **D1 Core has instability = 0.00** | Perfect stable foundation -- depends on nothing internal |
| **Clean WASM/Node binding pattern** | Consistent `-wasm` and `-node` suffix crates that depend only on their core |
| **RuVix sub-ecosystem is self-contained** | 25 crates with no outward dependencies except one edge to ruvector-coherence |
| **rvAgent sub-ecosystem is self-contained** | 9 crates forming a clean hierarchy with one edge out to ruvector-sona |
| **Workspace dependency centralization** | 36 key deps standardized in workspace Cargo.toml |

### 6.3 Recommendations

1. **Abstract ruvector-postgres**: Introduce trait-based storage interfaces in D1 Core so postgres does not need direct dependencies on attention, solver, or domain-expansion crates. Use dependency inversion.

2. **Expand workspace dependencies**: Add the next 15-20 most common external deps (dashmap, sha2, axum, ndarray, async-trait, futures, crossbeam, hex, base64, smallvec, etc.) to `[workspace.dependencies]` to prevent version drift.

3. **Protect ruvector-core API surface**: Given 29 dependents, consider a semver-locked public API with deprecation cycles. Add API compatibility tests.

4. **Standardize NPM testing**: Migrate the 2 jest-based packages to vitest. Remove the winston/pino duplication in @ruvector/agentic-integration.

5. **Consider quickcheck removal**: Both proptest (54 crates) and quickcheck are used for property testing. Standardize on proptest which has broader adoption in this workspace.

6. **Monitor ruvector-graph-transformer**: With deps spanning 5 domains, this is the highest cross-domain coupling point. Ensure integration test coverage is proportional.

---

## 7. Dependency Graph Summary Statistics

```
Workspace Composition:
  Rust crates (parsed):     165
  NPM packages:              55
  Total:                     220

Internal Dependencies (Rust):
  Edges:                     283
  Avg deps/crate:            1.7
  Max fan-in:                29 (ruvector-core)
  Max fan-out:               13 (mcp-brain-server)
  Circular dependencies:       0

External Dependencies (Rust):
  Unique crate deps:         232
  Workspace-managed:          36
  Locally-declared:          196

Cross-Domain Coupling:
  Cross-domain edges:         46
  Domains analyzed:           10
  Stable foundations (I<0.3):  3 (D1, D5, D3)
  High instability (I>0.7):   5 (D4, D8, D11, D12, D13)

NPM Dependencies:
  Unique prod deps:           40
  Unique dev deps:            29
  Internal dep edges:         12
```
