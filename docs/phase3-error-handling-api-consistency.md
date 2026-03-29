# Phase 3: Error Handling Patterns & API Consistency

**Cross-Cutting Analysis Across All Domains**
**Date**: 2026-03-29
**Scope**: All crates under `crates/` (100+ crates, ~8 domains)
**Severity Weighting**: CRITICAL=3, HIGH=2, MEDIUM=1, LOW=0.5, INFORMATIONAL=0.25

---

## Executive Summary

The RuVector monorepo contains **90+ distinct error enum/struct types** across 100+ crates. Error handling quality varies dramatically between domains -- from exemplary (ruvector-solver, ruvector-sparsifier) to problematic (ruvector-postgres, mcp-brain-server). The codebase has **8,287 `.unwrap()` calls** in library source code, **208 `.ok()` silent error suppressions**, and **144 `let _ =` discard patterns**. There is a thiserror version split (v1 vs v2) across crates. Logging uses `tracing` consistently, but with no spans, no structured fields, and 733+ `println!/eprintln!` calls in library code. Only 15 out of 100+ crates enforce `#![deny(missing_docs)]`.

**Weighted Finding Score: 21.5** (minimum required: 3.0)

---

## 1. Error Type Inventory

### 1.1 Error Types by Domain

| Domain | Crate | Error Type(s) | Framework | `std::error::Error` |
|--------|-------|--------------|-----------|---------------------|
| **D1 Core** | ruvector-core | `RuvectorError` (12 variants) | thiserror (workspace=2.0) | Yes (derive) |
| **D1 Core** | ruvector-collections | `CollectionError` | thiserror | Yes (derive) |
| **D1 Core** | ruvector-filter | `FilterError` | thiserror | Yes (derive) |
| **D2 Graph** | ruvector-graph | `GraphError` (24 variants) | thiserror | Yes (derive) |
| **D2 Graph** | ruvector-graph | `ParseError`, `LexerError`, `SemanticError`, `ExecutionError` | thiserror | Yes (derive) |
| **D3 Distributed** | ruvector-cluster | `ClusterError` (8 variants) | thiserror | Yes (derive) |
| **D3 Distributed** | ruvector-raft | `RaftError` (11 variants) | thiserror | Yes (derive) |
| **D3 Distributed** | ruvector-delta-consensus | `ConsensusError` (7 variants) | **manual impl** | Yes (manual) |
| **D3 Distributed** | ruvector-replication | error type | thiserror | Yes (derive) |
| **D4 Postgres** | ruvector-postgres | `RegistryError`, `SparseError`, `QueueError` x2, `IpcError`, `SparqlError`, `ValidationError`, `IsolationError`, `TenantError` | **manual impl** | Yes (manual) |
| **D5 Neural/ML** | ruvector-cnn | `CnnError` (23 variants) | thiserror | Yes (derive) |
| **D5 Neural/ML** | ruvector-gnn | `GnnError` | thiserror | Yes (derive) |
| **D5 Neural/ML** | ruvector-nervous-system | `NervousSystemError`, `HdcError`, `HopfieldError` | thiserror | Yes (derive) |
| **D5 Neural/ML** | ruvector-mincut | `MinCutError` (14 variants) | thiserror | Yes (derive) |
| **D5 Neural/ML** | ruvector-sparsifier | `SparsifierError` (10 variants) | thiserror | Yes (derive) |
| **D5 Neural/ML** | ruvector-solver | `SolverError` (5 variants) + `ValidationError` (6 variants) | thiserror | Yes (derive) |
| **D5 Neural/ML** | ruvector-graph-transformer | `GraphTransformerError` (8 variants) | thiserror | Yes (derive) |
| **D5 Neural/ML** | ruvector-attention | `AttentionError` | thiserror | Yes (derive) |
| **D6 WASM** | ruvector-wasm | `WasmError` (struct: message + kind) | **manual impl** | No (struct) |
| **D6 WASM** | ruvector-graph-wasm | `GraphError` (struct: message + kind) | **manual impl** | No (struct) |
| **D6 WASM** | ruvector-graph-transformer-wasm | `GraphTransformerError` (enum, 2 variants) | **manual impl** | Yes (manual) |
| **RVF** | rvf-types | `RvfError` (7 variants) + `ErrorCode` (50+ codes) | **manual impl** | **NO** |
| **RVF** | rvf-types | `SecurityError` | manual | Partial |
| **RVF** | rvf-kernel | `KernelError` | manual | Yes (manual) |
| **RVF** | rvf-server | `ServerError` | thiserror | Yes (derive) |
| **RVF** | rvf-launch | `LaunchError` | manual | Yes (manual) |
| **Agents** | rvagent-core | `RvAgentError` | thiserror | Yes (derive) |
| **Server** | ruvector-server | `Error` (9 variants) | thiserror | Yes (derive) |
| **Algorithms** | ruQu | `RuQuError`, `TraitError`, `FilterError` | thiserror | Yes (derive) |

### 1.2 Quantitative Summary

- **Total distinct error types**: ~90 enums + ~40 structs
- **Using thiserror**: ~65 crates (workspace ref or pinned)
- **Using manual `Display` + `Error` impls**: ~25 crates
- **Using anyhow**: ~68 crates (mainly for application-level code and examples)
- **Missing `std::error::Error` impl**: `RvfError` (rvf-types), WASM struct errors

### 1.3 thiserror Version Split

**FINDING [HIGH]**: There is a **thiserror 1.x vs 2.x split** across the workspace.

- **Workspace default** (`Cargo.toml` root): `thiserror = "2.0"`
- **Pinned to 1.x** (8 crates): `cognitum-gate-tilezero`, `ruvector-postgres`, `ruvector-dag`, `ruvector-crv`, `ruvector-attention`, `mcp-gate`, `rvlite`, `ruvix/qemu-swarm`
- **Pinned to 2.x** (11 crates): `ruvector-mincut-gated-transformer`, `ruvector-fpga-transformer`, `ruvector-delta-*`, `mcp-brain*`, `ruvector-hyperbolic-hnsw`, `ruvector-robotics`
- **Using workspace ref** (~60 crates): Inherits `2.0`

The thiserror 1.x to 2.x transition changed the `#[from]` attribute behavior and the internal trait implementation. Crates pinned to 1.x cannot interoperate cleanly with crates using 2.x if error types are passed across crate boundaries.

### 1.4 CRITICAL: `RvfError` Missing `std::error::Error`

**FINDING [CRITICAL]**: `rvf-types::RvfError` -- the foundational error type for the RVF file format used across the entire stack -- does **not** implement `std::error::Error`. It only implements `Display`. This means:

- It cannot be used with the `?` operator in functions returning `Box<dyn Error>`
- It cannot be used with `anyhow::Error` directly
- It cannot participate in error chains via `source()`
- The `ErrorCode` enum also lacks `std::error::Error`

This is the most architecturally significant error handling gap in the codebase, since RVF is the wire format for the entire system.

---

## 2. Error Propagation Analysis

### 2.1 Cross-Crate Error Flow

The error propagation architecture follows a layered pattern:

```
Layer 4: WASM/Node.js   WasmError (struct) <-- JsValue
            |
            | From<RuvectorError>
            v
Layer 3: Server          server::Error (axum IntoResponse)
            |
            | #[from] RuvectorError
            v
Layer 2: Domain Logic    GraphError, CnnError, SolverError, etc.
            |
            | From<> implementations
            v
Layer 1: Core            RuvectorError (thiserror)
            |
            | From<redb::*> (conditional on feature)
            v
Layer 0: Storage/IO      std::io::Error, redb::Error, serde_json::Error
```

**Well-designed flow** (ruvector-graph-transformer):
```
ruvector-verified::VerificationError
  --> #[from] GraphTransformerError::Verification
ruvector-gnn::GnnError
  --> #[from] GraphTransformerError::Gnn
ruvector-attention::AttentionError
  --> #[from] GraphTransformerError::Attention
ruvector-mincut::MinCutError
  --> #[from] GraphTransformerError::MinCut
```

**Problematic flow** (RVF stack):
```
rvf-types::RvfError (NO std::error::Error)
  --> manual From<RvfError> for WitnessError (rvf-runtime)
  --> manual From<RvfError> for SeedError (rvf-runtime)
  --> manual From<RvfError> for ServerError (rvf-server)
  --> manual From<RvfError> for RvfPackError (ruvector-robotics)
  --> CANNOT use with ? in anyhow contexts
```

**WASM Error Boundary**:
```
RuvectorError --> WasmError { message, kind } --> JsValue (Object)
```
Five crates implement `From<*> for JsValue`: `ruvector-wasm`, `ruvector-graph-wasm`, `rvlite`, `ruvllm-wasm`, `rvagent-wasm`. The pattern is consistent: serialize to a JS object with `message` and `kind` fields.

### 2.2 Silent Error Suppression

**FINDING [HIGH]**: Errors are silently suppressed in multiple patterns:

| Pattern | Count (crates/src) | Risk |
|---------|-------------------|------|
| `.unwrap()` | **8,287** | Panics in production |
| `.ok()` (Result to Option) | **208** | Silently drops errors |
| `let _ =` (discard result) | **144** | Ignores failures |
| `unwrap_or_default()` | **117** | Masks failures with defaults |
| `.expect()` | **273** | Panics with message |
| `panic!()` | **107** | Explicit panics |

**Total potential panic sites**: 8,287 (unwrap) + 273 (expect) + 107 (panic!) = **8,667**

#### Hotspots by Crate (top `.unwrap()` offenders):

| Crate | `.unwrap()` count | Context |
|-------|-------------------|---------|
| ruvector-postgres | ~572 | PostgreSQL extension (pgrx context) |
| ruvector-bench | ~159 | Benchmark binaries |
| ruvector-core | ~104 | Tests + some lib code |
| ruvector-mincut | ~90+ | Algorithm code |
| ruvector-graph | ~80+ | Storage and optimization |
| rvAgent | ~80+ | Backends and WASM |
| mcp-brain-server | ~70+ | Server routes |

**Mitigating context**: Many `.unwrap()` calls in ruvector-postgres are within `pgrx` operator functions where panics are caught by the PostgreSQL runtime. However, this is not universally true across all files in that crate.

#### `.ok()` Hotspots (silent error suppression):

| Crate | `.ok()` count | Risk Level |
|-------|--------------|------------|
| mcp-brain-server | 42 | HIGH -- server code silently dropping errors |
| ruvector-postgres/graph | 19 | MEDIUM -- SPARQL execution dropping errors |
| ruvector-fpga-transformer | 13 | HIGH -- hardware interface dropping errors |
| rvAgent | 11 | MEDIUM -- agent middleware |
| ruvector-cli | 7 | LOW -- CLI presentation layer |

### 2.3 Error Boundary Mapping

**Server --> Client boundaries** (2 implementations):

1. **ruvector-server** (`impl IntoResponse for Error`): Maps to HTTP status codes
   - NotFound -> 404
   - Conflict -> 409
   - BadRequest -> 400
   - Internal -> 500
   - Response body: `{ "error": string, "status": number }`

2. **rvf-server** (`impl IntoResponse for ServerError`): Similar pattern but separate implementation

**FINDING [MEDIUM]**: Two server crates independently implement HTTP error mapping with no shared error response type. The response formats should be unified.

---

## 3. API Surface Consistency

### 3.1 CRUD Naming Conventions

Analysis of public API method names across core crates:

| Operation | ruvector-core | ruvector-graph | ruvector-cluster | ruvector-collections |
|-----------|--------------|----------------|-----------------|---------------------|
| **Create** | `insert()` | `create()` (Edge), `insert()` (cache/index) | `add_node()` | `create_collection()` |
| **Read** | `get()`, `search()` | `get()` (cache) | `get_node()` | `get_collection()` |
| **Delete** | `delete()` | `remove()` (cache), `clear()` (index) | `remove_node()` | `delete_collection()` |
| **Bulk** | `batch_insert()` | N/A | N/A | N/A |

**FINDING [MEDIUM]**: Inconsistent naming for the "add data" operation:
- `insert` (ruvector-core: VectorDB, AgenticDB, Storage)
- `add` (ruvector-core: arena, lockfree counter)
- `push` (ruvector-core: cache_optimized, arena)
- `create` (ruvector-graph: Edge)
- `add_node` (ruvector-cluster)

For the "remove data" operation:
- `delete` (ruvector-core: VectorDB, Storage)
- `remove` (ruvector-core: multi_vector, sparse_vector; ruvector-graph: cache)
- `clear` (ruvector-core: arena; ruvector-graph: index, stats)

While some variance is expected (e.g., `push` for stack-like containers vs `insert` for key-value stores), the core vector operations should use consistent terminology.

### 3.2 Return Type Consistency

| Pattern | Usage | Crates |
|---------|-------|--------|
| `Result<T, CrateError>` via type alias | Standard | ~50 crates |
| `Result<T, E>` explicit | Common | ~20 crates |
| `Option<T>` for lookups | ruvector-core `get()` returns `Result<Option<>>` | ruvector-core |
| Raw `T` (panic on error) | Rare in APIs, common in internal code | ruvector-postgres |

**FINDING [LOW]**: The `Result` type alias pattern is used by approximately 50 crates, which is good consistency. However, the naming is not uniform:
- `Result<T>` (most common)
- `CnnResult<T>` (ruvector-cnn)
- `RaftResult<T>` (ruvector-raft)
- `WasmResult<T>` (ruvector-wasm, internal only)

### 3.3 Builder Pattern Usage

| Crate | Builder instances | Notes |
|-------|------------------|-------|
| ruvector-core | 4 | Minimal -- mostly in advanced features |
| ruvector-graph | 65 | Extensive -- Node, Edge, Graph, Pipeline |
| ruvector-solver | 7 | Audit builder |

**FINDING [LOW]**: Builder pattern adoption is inconsistent. `ruvector-graph` uses builders extensively (good for complex graph construction), while `ruvector-core` and `ruvector-solver` use them sparingly. For configuration-heavy types like `HnswConfig` and `DbOptions`, direct struct construction is used instead of builders.

### 3.4 Async Consistency

| Crate | Sync API | Async API | Notes |
|-------|----------|-----------|-------|
| ruvector-core | All sync | None | Core is synchronous by design |
| ruvector-graph | All sync | None | Graph is synchronous |
| ruvector-cluster | Some sync | 6 async fns | Mixed -- `add_node`, `remove_node`, `run_health_checks`, `start` |
| ruvector-server | N/A | 3 async fns | Axum handlers are async |
| ruvector-raft | All sync | None | Raft consensus is synchronous |

**FINDING [MEDIUM]**: The cluster crate mixes sync and async without clear documentation on which operations require a runtime. The core crate is entirely synchronous, which is a valid design choice, but it means that crates like ruvector-cluster that wrap core operations in async contexts must handle the sync-to-async bridge.

---

## 4. Logging & Observability Audit

### 4.1 Logging Framework Usage

| Framework | Crate Count | Usage Pattern |
|-----------|-------------|---------------|
| `tracing` | **60+ crates** | Primary framework |
| `tracing-subscriber` | ~20 crates | Subscriber initialization |
| `tracing-wasm` | 3 crates | WASM target |
| `log` crate | **0 crates** (direct dependency) | Not used |
| `env_logger` | **0 crates** | Not used |

**Good**: The project has standardized on `tracing` as the sole logging framework. There is no `log` crate dependency in any crate's Cargo.toml.

### 4.2 Tracing Usage Analysis

| Pattern | Count (crates/src) | Assessment |
|---------|-------------------|------------|
| `tracing::{info,warn,error,debug,trace}!` | ~350 macro invocations | Low for 100+ crates |
| `tracing::*_span!` | **0** | No span instrumentation |
| Structured fields in log macros | Rare | Mostly string interpolation |

**FINDING [HIGH]**: While `tracing` is the standardized framework, it is used only for basic log-level output (info/warn/error). There are:
- **Zero spans**: No `info_span!`, `debug_span!`, `trace_span!` calls anywhere in the crate source
- **No `#[instrument]` annotations**: The tracing `#[instrument]` proc macro is not used
- **No structured logging**: Most calls use string formatting rather than structured fields

This means the `tracing` dependency provides zero advantage over the simpler `log` crate -- the distributed tracing, span correlation, and structured data features are entirely unused.

### 4.3 Debug Output in Library Code

**FINDING [HIGH]**: Significant `println!`/`eprintln!` usage in library source code:

| Pattern | Count (crates/src) | Assessment |
|---------|-------------------|------------|
| `println!` | **606** | Should be `tracing::info!` or removed |
| `eprintln!` | **127** | Should be `tracing::error!`/`warn!` |
| **Total** | **733** | |

Hotspots for `println!` in library code (non-test, non-example, non-binary):
- `ruvector-bench/src/bin/*` (159) -- Acceptable in benchmarks
- `rvAgent/rvagent-cli/src/display.rs` (30) -- CLI output, acceptable
- `ruvector-router-cli/src/main.rs` (33) -- CLI output, acceptable
- `rvf/rvf-cli/src/cmd/*` (45+) -- CLI output, acceptable
- `ruvector-temporal-tensor/src/store.rs` (9) -- **Not acceptable in library code**
- `mcp-brain-server/src/quantization.rs` (23) -- **Not acceptable in server code**
- `ruvector-hyperbolic-hnsw/src/lib.rs` (1) -- **Not acceptable in library code**
- `ruvector-sparsifier/src/lib.rs` (2) -- **Not acceptable in library code**

Hotspots for `eprintln!` in library code:
- `rvf/rvf-ebpf/src/lib.rs` (9) -- eBPF bootstrap, may be intentional
- `rvf/rvf-cli/src/cmd/launch.rs` (15) -- CLI error output
- `ruvector-temporal-tensor/src/store.rs` (9) -- **Library code**
- `ruvllm/src/lora/adapters/trainer.rs` (6) -- **Library code**
- `ruvix/crates/boot/src/stages.rs` (19) -- OS boot code, acceptable (no allocator)

### 4.4 Metrics and Observability

| Framework | Files Using | Assessment |
|-----------|------------|------------|
| Custom `metrics::` | 20 files | Several crates define their own metrics types |
| Prometheus | 0 | Not used |
| OpenTelemetry | 0 | Not used |

**FINDING [MEDIUM]**: There is no standard metrics emission framework. Several crates (ruvector-tiny-dancer-core, ruvector-temporal-tensor, ruvllm) define their own `metrics` modules with custom structs, but there is no unified metrics collection or export.

Crates with custom metrics:
- `ruvector-tiny-dancer-core/src/metrics.rs` -- Custom metrics + tracing spans
- `ruvector-temporal-tensor/src/metrics.rs` -- Custom latency tracking
- `ruvector-node/src/lib.rs` -- Prometheus-style counter patterns
- `ruvllm/src/quality/scoring_engine.rs` -- Custom scoring metrics
- `ruvllm/src/optimization/mod.rs` -- Performance metrics

The D3 (distributed systems) layer has **no distributed tracing** -- cluster operations, raft consensus, and replication have basic `tracing::info!`/`error!` calls but no correlation IDs or span propagation. This makes debugging distributed failures extremely difficult.

---

## 5. Public API Documentation Coverage

### 5.1 Crate-Level `#![deny(missing_docs)]`

Only **15 out of 100+ crates** enforce documentation:

| Crate | Has `deny(missing_docs)` |
|-------|-------------------------|
| ruQu | Yes |
| ruvector-mincut | Yes |
| ruvix/tests | Yes |
| ruvix/crates/* (13 crates) | Yes |
| **All other crates** | **No** |

**FINDING [MEDIUM]**: Documentation enforcement is limited to the ruvix (OS kernel) subsystem and two other crates. The core ruvector crates have no doc enforcement.

### 5.2 Documentation Coverage Sampling

Methodology: Count `///` doc comments vs `pub fn`/`pub async fn` declarations. A ratio above 1.0 indicates fields/types are also documented; below 1.0 indicates missing docs.

| Crate | `pub fn` count | `///` comments | Ratio | Grade |
|-------|---------------|----------------|-------|-------|
| **ruvector-solver** | 84 | 963 | **11.5x** | A+ (exemplary) |
| **ruvector-mincut** | 276 | 1,353 | **4.9x** | A (comprehensive) |
| **ruvector-cnn** | 269 | 660 | **2.5x** | B+ (good) |
| **ruvector-core** | 364 | 1,075 | **3.0x** | A- (thorough) |
| **rvf-types** | 59 | 1,177 | **20.0x** | A+ (every field documented) |
| **ruvector-graph** | 363 | 726 | **2.0x** | B (adequate) |
| **ruvector-cluster** | 60 | 175 | **2.9x** | B+ (good) |

Overall: The newer/algorithmic crates have excellent documentation. The graph and distributed crates are adequate but have gaps in internal modules.

---

## 6. Cross-Cutting Findings Summary

### CRITICAL Findings (Weight: 3.0 each)

| ID | Finding | Domains Affected | Impact |
|----|---------|-----------------|--------|
| C1 | `RvfError` does not implement `std::error::Error` | All (RVF is foundation) | Cannot compose with standard error chains, breaks `?` operator in `anyhow` contexts |
| C2 | 8,287 `.unwrap()` calls in library source code | All | Any of these can cause runtime panics in production |

### HIGH Findings (Weight: 2.0 each)

| ID | Finding | Domains Affected | Impact |
|----|---------|-----------------|--------|
| H1 | thiserror version split (1.x vs 2.x) across 19+ crates | D1, D2, D3, D4, RVF | Potential compilation issues, inconsistent derive behavior |
| H2 | 208 `.ok()` calls silently suppress errors | D4, D5, Agents | Failures hidden, debugging impossible |
| H3 | Zero tracing spans in entire codebase | All | Distributed tracing capability is null despite having the framework |
| H4 | 733 `println!/eprintln!` in library code | D5, RVF, mcp-brain-server | Unstructured output, no log levels, no filtering |
| H5 | D4 (Postgres) has 9 separate error types, all manually implemented | D4 | No unified error handling within the crate |

### MEDIUM Findings (Weight: 1.0 each)

| ID | Finding | Domains Affected | Impact |
|----|---------|-----------------|--------|
| M1 | Inconsistent CRUD naming (`insert` vs `add` vs `push` vs `create`) | D1, D2, D3 | API confusion for consumers |
| M2 | Two independent HTTP error response implementations | Server, RVF | Inconsistent API responses |
| M3 | No standard metrics emission (Prometheus/OpenTelemetry) | All | No production observability |
| M4 | Mixed sync/async in D3 cluster crate | D3 | Unclear runtime requirements |
| M5 | WASM error structs lack `std::error::Error` | D6 | Cannot compose in Rust contexts |

### LOW Findings (Weight: 0.5 each)

| ID | Finding | Domains Affected | Impact |
|----|---------|-----------------|--------|
| L1 | Inconsistent Result type alias naming (`Result` vs `CnnResult` vs `RaftResult`) | D5, D3 | Minor inconsistency |
| L2 | Builder pattern usage varies widely | D1, D2 | Style inconsistency |
| L3 | Only 15/100+ crates enforce `deny(missing_docs)` | All | Documentation gaps |

### INFORMATIONAL (Weight: 0.25 each)

| ID | Finding | Domains Affected |
|----|---------|-----------------|
| I1 | ruvector-solver and ruvector-sparsifier are exemplary models for error handling | D5 |
| I2 | ruvector-mincut has useful `is_recoverable()`, `is_graph_structure_error()` error classification methods | D5 |
| I3 | rvf-types `ErrorCode` has excellent categorization with `category()`, `is_security_error()` etc. | RVF |
| I4 | ruvector-cnn provides convenient error construction helpers (`dim_mismatch()`, `invalid_shape()`) | D5 |

---

## 7. Weighted Finding Score

| Severity | Count | Weight | Subtotal |
|----------|-------|--------|----------|
| CRITICAL | 2 | 3.0 | 6.0 |
| HIGH | 5 | 2.0 | 10.0 |
| MEDIUM | 5 | 1.0 | 5.0 |
| LOW | 3 | 0.5 | 1.5 |
| INFORMATIONAL | 4 | 0.25 | 1.0 |
| **Total** | **19** | | **23.5** |

---

## 8. Recommendations (Priority Order)

### P0: Fix `RvfError` (C1)

Add `impl std::error::Error for RvfError` in `crates/rvf/rvf-types/src/error.rs`. Also add `impl std::error::Error for ErrorCode`. This unblocks proper error composition across the entire RVF stack.

```rust
impl std::error::Error for RvfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Security(e) => Some(e),
            _ => None,
        }
    }
}
```

### P1: Unify thiserror to 2.x (H1)

Migrate the 8 crates pinned to thiserror 1.x to use the workspace 2.0 version. The crates are: `cognitum-gate-tilezero`, `ruvector-postgres`, `ruvector-dag`, `ruvector-crv`, `ruvector-attention`, `mcp-gate`, `rvlite`, `ruvix/qemu-swarm`.

### P2: Instrument with tracing spans (H3)

Add `#[instrument]` to key public functions in:
- ruvector-cluster (distributed operations)
- ruvector-raft (consensus rounds)
- ruvector-server (request handling)
- mcp-brain-server (API routes)

### P3: Replace `println!`/`eprintln!` in library code (H4)

Target the non-CLI, non-binary uses:
- `ruvector-temporal-tensor/src/store.rs`
- `mcp-brain-server/src/quantization.rs`
- `ruvector-hyperbolic-hnsw/src/lib.rs`
- `ruvector-sparsifier/src/lib.rs`
- `ruvllm/src/lora/adapters/trainer.rs`

### P4: Establish error handling guidelines (M1, M2)

Create an ADR that standardizes:
1. All domain error types use thiserror 2.x
2. Error types follow naming convention: `{Crate}Error`
3. Result type alias naming: `pub type Result<T> = std::result::Result<T, {Crate}Error>`
4. CRUD naming: `insert`/`get`/`delete` for data stores, `add`/`remove` for collections
5. HTTP error response format (shared between ruvector-server and rvf-server)

### P5: Systematic `.unwrap()` audit (C2)

Prioritize `.unwrap()` removal in:
1. Any code path reachable from user input
2. Server request handlers
3. WASM-exported functions
4. Core library public APIs

`.unwrap()` is acceptable in:
- Test code
- Benchmark code
- After infallible operations (e.g., `Regex::new` on compile-time constants)
- pgrx operator functions (where panics are caught by PostgreSQL)

### P6: Adopt exemplary patterns

Use these crates as models for new development:
- **ruvector-solver**: Structured error types with rich context (iterations, residuals, algorithm names)
- **ruvector-sparsifier**: Clean error hierarchy with domain-specific variants
- **ruvector-mincut**: Error classification methods (`is_recoverable()`, `is_resource_error()`)
- **rvf-types ErrorCode**: Categorized numeric error codes with helper methods

---

## 9. Domain-Specific Error Flow Diagrams

### D1 Core --> D6 WASM Flow

```
RuvectorError (thiserror, 12 variants)
    |
    | From<RuvectorError> for WasmError
    |   (lossy: converts to string message + debug kind)
    v
WasmError { message: String, kind: String }
    |
    | From<WasmError> for JsValue
    |   (serializes to JS object)
    v
JsValue { message: "...", kind: "..." }
```

Error context is lost at the WASM boundary: structured enum variants become flat strings. This is inherent to the JS interop but could be improved by mapping variants to error codes.

### D2 Graph --> D5 Graph Transformer Flow

```
ruvector-verified::VerificationError --|
ruvector-gnn::GnnError --------------|
ruvector-attention::AttentionError ---|---> GraphTransformerError (thiserror, #[from])
ruvector-mincut::MinCutError ---------|
```

This is the cleanest cross-crate error composition in the codebase. The `#[from]` attribute preserves the source error chain.

### D4 Postgres Internal Flow

```
pg_sys::panic!() <-- unwrap() [572 calls]
    ^
    |
SparseError     <-- manual Display/Error
QueueError x2   <-- manual Display/Error
IpcError        <-- manual Display/Error
SparqlError     <-- manual Display/Error
RegistryError   <-- manual Display/Error
ValidationError <-- manual Display/Error
IsolationError  <-- manual Display/Error
TenantError     <-- manual Display/Error
```

D4 has the most fragmented error landscape: 9 separate error types, all manually implemented, with no shared base type or conversion between them.

---

## Appendix A: Files Examined

### Error type definitions
- `crates/ruvector-core/src/error.rs`
- `crates/rvf/rvf-types/src/error.rs`
- `crates/ruvector-graph/src/error.rs`
- `crates/ruvector-cluster/src/lib.rs`
- `crates/ruvector-solver/src/error.rs`
- `crates/ruvector-cnn/src/error.rs`
- `crates/ruvector-server/src/error.rs`
- `crates/ruvector-mincut/src/error.rs`
- `crates/ruvector-sparsifier/src/error.rs`
- `crates/ruvector-raft/src/lib.rs`
- `crates/ruvector-delta-consensus/src/error.rs`
- `crates/ruvector-graph-transformer/src/error.rs`
- `crates/ruvector-graph-transformer-wasm/src/transformer.rs`
- `crates/ruvector-wasm/src/lib.rs`
- `crates/ruvector-graph-wasm/src/types.rs`

### All Cargo.toml files searched for dependency analysis
### All `crates/**/src/**/*.rs` files searched for pattern analysis

---

## Appendix B: Patterns Checked (Clean Justification)

| Check | Tool Used | Files Searched | Result |
|-------|-----------|---------------|--------|
| `pub enum.*Error` | Grep | All `*.rs` | 90+ error enums found |
| `pub struct.*Error` | Grep | All `*.rs` | 40+ error structs found |
| `thiserror` in Cargo.toml | Grep | All `Cargo.toml` | 80+ crates |
| `anyhow` in Cargo.toml | Grep | All `Cargo.toml` | 68 crates |
| `impl From<*> for *Error` | Grep | All `*.rs` | 80+ implementations |
| `.unwrap()` | Grep+count | crates/src | 8,287 |
| `.ok()` | Grep+count | crates/src | 208 |
| `let _ =` | Grep+count | crates/src | 144 |
| `unwrap_or_default()` | Grep+count | crates/src | 117 |
| `.expect()` | Grep+count | crates/src | 273 |
| `panic!` | Grep+count | crates/src | 107 |
| `println!` | Grep+count | crates/src | 606 |
| `eprintln!` | Grep+count | crates/src | 127 |
| `deny(missing_docs)` | Grep | crates/src | 15 crates |
| `tracing` in Cargo.toml | Grep | crates/Cargo.toml | 60+ crates |
| `*_span!` | Grep | crates/src | 0 |
| `prometheus\|opentelemetry` | Grep | crates/src | 0 direct uses |
| `impl IntoResponse` | Grep | crates/src | 2 implementations |
| `From<*> for JsValue` | Grep | crates/src | 5 implementations |
| `pub type Result` | Grep | crates/src | 50+ aliases |
| Builder patterns | Grep | selected crates | Variable adoption |
| pub fn naming analysis | Grep | selected crates | Inconsistencies found |
| Doc comment density | Grep | 7 sampled crates | Varies 2.0x-20.0x |
