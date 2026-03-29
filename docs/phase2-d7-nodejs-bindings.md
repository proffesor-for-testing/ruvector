# Phase 2 Deep Quality Analysis -- Domain 7: Node.js Bindings

**Priority**: P2 MEDIUM
**Date**: 2026-03-29
**Analyst**: QE Integration Tester (V3)
**Scope**: All `*-node` crates (10), `npm/packages/*` directories (56 packages)

---

## 1. Inventory of Node.js Binding Crates and NPM Packages

### 1.1 Node Binding Crates (Rust -> JS via NAPI-RS)

| Crate | Wraps | LOC | Files | State |
|-------|-------|-----|-------|-------|
| `agentic-robotics-node` | `agentic-robotics-core` (pub/sub) | 233 | 1 | Complete |
| `ruvector-attention-node` | `ruvector-attention` (7 attention variants, training, async) | 2,515 | 5 | Complete |
| `ruvector-gnn-node` | `ruvector-gnn` (GNN layer, tensor compression, search) | 421 | 1 | Complete |
| `ruvector-graph-node` | `ruvector-graph` + `ruvector-core` (graph DB, hypergraph, Cypher) | 1,060 | 4 | Complete |
| `ruvector-graph-transformer-node` | Self-contained transformer (proof-gated, physics, bio, temporal, economic) | 2,151 | 2 | Complete |
| `ruvector-mincut-brain-node` | `ruvector-mincut` (WASM stubs for pi.ruv.io brain) | 59 | 1 | Complete (WASM only, not NAPI) |
| `ruvector-mincut-node` | `ruvector-mincut` (dynamic min-cut, hierarchy, local k-cut) | 545 | 1 | Complete |
| `ruvector-node` | `ruvector-core` + collections + filter + metrics (vector DB) | 779 | 1 | Complete |
| `ruvector-solver-node` | `ruvector-solver` (sparse linear systems, PageRank) | 1,182 | 1 | Complete |
| `ruvector-tiny-dancer-node` | `ruvector-tiny-dancer-core` (neural routing) | 286 | 1 | Complete |

**Total Rust binding code**: 9,231 LOC across 18 source files.

### 1.2 NPM Packages (56 total)

#### NAPI-RS Native Binding Packages (published to npm)

| Package | Version | Wraps Crate | Has .d.ts | Has Tests |
|---------|---------|-------------|-----------|-----------|
| `@ruvector/core` | 0.1.30 | `ruvector-node` | Yes (29 lines) | No JS tests |
| `@ruvector/graph-node` | 2.0.2 | `ruvector-graph-node` | Yes (370 lines) | No |
| `@ruvector/tiny-dancer` | 0.1.17 | `ruvector-tiny-dancer-node` | Yes (138 lines) | No |
| `@ruvector/router` | 0.1.28 | (native router crate) | Yes (284 lines) | No |
| `@ruvector/rvf-node` | 0.1.7 | (RVF native crate) | Yes (95 lines) | No |
| `@ruvector/attention` | 0.1.4 | `ruvector-attention-node` | (in crate dir) | No |
| `@ruvector/sona` | (in npm) | (native crate) | (in npm) | No |
| `@ruvector/rvdna` | (in npm) | (native crate) | (in npm) | No |

#### Pure TypeScript/JS Packages

| Package | Version | Test Framework | Has Tests |
|---------|---------|----------------|-----------|
| `@ruvector/agentic-synth` | 0.1.6 | vitest | 6 test files |
| `@ruvector/agentic-integration` | 1.0.0 | jest | 0 test files found |
| `ruvbot` | 0.3.1 | vitest | (integration tests) |
| `ruvector` (CLI) | 0.2.19 | none | No |
| `@ruvector/node` | 0.1.22 | none | No |
| `@ruvector/rvf` | 0.2.0 | jest | No |
| `@ruvector/cli` | 0.1.28 | none | No |
| `@ruvector/pi-brain` | 0.1.0 | none | No |
| `@ruvector/rvf-mcp-server` | 0.1.3 | none | No |
| `@ruvector/ospipe` | 0.1.2 | none | No |
| `@ruvector/ruvector-extensions` | (in npm) | vitest | 4 test files |
| `@ruvector/ruvllm` | (in npm) | node:test | 3 test files |

#### Platform-Specific Binary Packages (15 total)

Packages: `router-{darwin-arm64,darwin-x64,linux-arm64-gnu,linux-x64-gnu,win32-x64-msvc}`, `ruvllm-{same 5}`, `tiny-dancer-{same 5}`.

#### Special Case: `@ruvector/spiking-neural`

Uses **node-gyp** + **node-addon-api** (C++ SIMD), NOT napi-rs. This is the only non-Rust native binding. No TypeScript definitions.

---

## 2. NAPI Safety Audit

### 2.1 NAPI Library & Version

All 9 NAPI crates (excluding `ruvector-mincut-brain-node` which is WASM) use **napi-rs v2.16** with `napi9` feature flag. The workspace Cargo.toml declares:

```toml
napi = { version = "2.16", default-features = false, features = ["napi9", "async", "tokio_rt"] }
napi-derive = "2.16"
```

**Finding (INCONSISTENCY)**: `ruvector-attention-node` declares `napi = { version = "2" }` (loose) while `ruvector-graph-transformer-node` declares `napi = { version = "2.16" }` (pinned). Both override workspace defaults. Other crates use `napi = { workspace = true }`. This version skew is low-risk today but could cause ABI issues in future updates.

**Severity**: LOW
**Recommendation**: Unify all crates to use `napi = { workspace = true }`.

### 2.2 Unsafe Blocks

| Crate | Unsafe Count | Details |
|-------|-------------|---------|
| `ruvector-mincut-brain-node` | 1 | `unsafe extern "C" fn feature_extract()` -- WASM V1 ABI stub, documented, no-op body. Acceptable. |
| All other 9 crates | 0 | Zero unsafe blocks. |

**Verdict**: Excellent. The NAPI-RS `#[napi]` macro generates all FFI glue safely. The single `unsafe` is a justified WASM export stub that performs no memory operations.

### 2.3 Type Conversion (Rust -> JS)

The binding crates use consistent, safe patterns:

| Rust Type | JS Type | Conversion Pattern |
|-----------|---------|-------------------|
| `Vec<f32>` | `Float32Array` | `Float32Array::new(vec)` -- zero-copy when possible |
| `Vec<f64>` | `Array<number>` | Direct via napi serde |
| `String` | `string` | Direct |
| `serde_json::Value` | `any` (JSON) | `serde_json::to_value()` / `from_value()` |
| `HashMap<K,V>` | `object` | Via `#[napi(object)]` derive |
| Custom structs | TypeScript class/object | `#[napi]` struct or `#[napi(object)]` |
| `Result<T>` | throws `Error` | `Error::from_reason()` or `Error::new(Status, msg)` |

**Finding (SERIALIZATION OVERHEAD)**: `ruvector-graph-transformer-node` passes almost all complex return types as `serde_json::Value` (mapped to `any` in TypeScript), which means every call goes through JSON serialization/deserialization. This adds overhead and loses type safety on the JS side. The `ruvector-gnn-node` crate is better -- it uses `TensorCompress.compress()` returning a JSON string explicitly, which is at least transparent about the cost.

**Severity**: MEDIUM
**Impact**: Performance and type safety for graph-transformer-node.

### 2.4 Thread Safety

| Crate | Concurrency Pattern | Risk |
|-------|-------------------|------|
| `agentic-robotics-node` | `Arc<RwLock<HashMap>>` + tokio async | Safe |
| `ruvector-attention-node` | No shared state, stateless structs | Safe |
| `ruvector-gnn-node` | No shared state | Safe |
| `ruvector-graph-node` | `Arc<RwLock<...>>` x5 + `tokio::task::spawn_blocking` | See below |
| `ruvector-graph-transformer-node` | `&mut self` on `GraphTransformer` | Safe (NAPI enforces single-threaded access to `&mut self`) |
| `ruvector-mincut-node` | `Arc<Mutex<DynamicMinCut>>` | See below |
| `ruvector-node` | `Arc<RwLock<CoreVectorDB>>` | See below |
| `ruvector-solver-node` | Stateless functions + `spawn_blocking` | Safe |
| `ruvector-tiny-dancer-node` | `parking_lot::RwLock` (not std) | Safe |

No `!Send`/`!Sync` types are exposed across thread boundaries. All shared state uses `Arc<RwLock>` or `Arc<Mutex>`, which is correct.

### 2.5 Memory Management

NAPI-RS handles GC integration automatically via its `Reference` system. When a JS object wrapping a `#[napi]` Rust struct is garbage collected, the Rust `Drop` implementation runs. All binding crates rely on this default behavior, which is correct.

**No manual Drop implementations** were found in any binding crate. This means Rust's automatic `Drop` handles cleanup for `Arc`, `RwLock`, `HashMap`, etc.

**Finding (NO EXPLICIT CLEANUP API)**: `ruvector-graph-node` creates persistent storage (`GraphStorage`) with file handles, but provides no explicit `close()` method. Cleanup happens only on GC, which is non-deterministic. If the JS process exits abruptly, storage may not be flushed.

**Severity**: MEDIUM
**Recommendation**: Add explicit `close()` / `dispose()` methods to `GraphDatabase` and `VectorDB` for deterministic cleanup.

### 2.6 Error Propagation

All binding crates convert Rust errors to proper JS `Error` objects via:
- `Error::from_reason(format!("..."))` -- most common pattern
- `Error::new(Status::InvalidArg, msg)` -- for validation errors
- `.map_err(|e| Error::from_reason(e.to_string()))` -- standard pattern

**No silent failures detected.** All error paths produce a JS-visible error.

### 2.7 Panic Risk (unwrap/expect in Non-Test Code)

| Crate | unwrap/expect Count | Pattern |
|-------|-------------------|---------|
| `ruvector-graph-transformer-node` | 27 | Mostly in `transformer.rs` (internal implementation) |
| `ruvector-graph-node` | 19 | `expect("RwLock poisoned")` in async closures |
| `ruvector-node` | 14 | `expect("RwLock poisoned")` in async closures |
| `ruvector-mincut-node` | 9 | `self.inner.lock().unwrap()` (Mutex) |
| `ruvector-solver-node` | 3 | Minor |

**Finding (CRITICAL -- PANIC ON POISONED LOCK)**: `ruvector-graph-node` has 15 instances of `expect("RwLock poisoned")` inside `spawn_blocking` closures. If any prior operation panics and poisons the lock, all subsequent operations will also panic, crashing the Node.js process. `ruvector-mincut-node` has 8 instances of `self.inner.lock().unwrap()` which have the same problem.

**Severity**: HIGH
**Impact**: A single panicking operation can cascade into process crash for all subsequent callers.
**Recommendation**: Replace `lock().unwrap()` and `write().expect(...)` with proper error handling:
```rust
let guard = self.inner.lock()
    .map_err(|e| Error::from_reason(format!("Lock poisoned: {}", e)))?;
```

---

## 3. TypeScript Definitions

### 3.1 Coverage

| Package | .d.ts Present | Generated By | Quality |
|---------|--------------|-------------|---------|
| `@ruvector/core` | Yes (29 lines) | Hand-written | Minimal but correct |
| `@ruvector/graph-node` | Yes (370 lines) | NAPI-RS auto-gen | Good, 1 `any` occurrence |
| `@ruvector/tiny-dancer` | Yes (138 lines) | NAPI-RS auto-gen | Good, zero `any` |
| `@ruvector/router` | Yes (284 lines) | NAPI-RS auto-gen | Good, zero `any` |
| `@ruvector/rvf-node` | Yes (95 lines) | NAPI-RS auto-gen | Good, zero `any` |
| `@ruvector/attention` (in crate) | (no index.d.ts in crate) | Not found | **MISSING** |
| `ruvector-graph-transformer-node` (in crate) | Yes (461 lines) | NAPI-RS auto-gen | Poor -- 14 `any` types |
| `@ruvector/spiking-neural` | **No** | N/A | **MISSING** |

### 3.2 `any` Type Usage

**Finding (TYPE SAFETY GAP)**: `ruvector-graph-transformer-node/index.d.ts` has **14 occurrences of `any`**:
- Constructor parameter: `config?: any | undefined | null`
- Return types for 10+ methods: `createProofGate()`, `proveDimension()`, `composeProofs()`, `sublinearAttention()`, `hamiltonianStep()`, `hamiltonianStepGraph()`, `spikingStep()`, `verifiedStep()`, `verifiedTrainingStep()`, `productManifoldAttention()`, `grangerExtract()`, `gameTheoreticAttention()`, `stats()`
- Input parameters for edge arrays: `Array<any>`

This is because the Rust code passes `serde_json::Value` for all complex types. NAPI-RS maps this to `any` in TypeScript.

**Severity**: MEDIUM
**Impact**: No compile-time type checking for the most feature-rich binding.
**Recommendation**: Define explicit TypeScript interfaces for each return type and use `@napi-rs/cli`'s override mechanism or a hand-written wrapper.

### 3.3 Definition Accuracy

The `@ruvector/core` hand-written `index.d.ts` (29 lines) is **incomplete relative to the Rust API**:
- Missing: `CollectionManager`, `getMetrics()`, `getHealth()`, filter types, HNSW config types, quantization types
- The Rust `ruvector-node` crate exposes ~15 classes/functions; the `.d.ts` only declares `VectorDb` + 3 interfaces
- `hnswConfig` typed as `any` in the constructor

**Severity**: MEDIUM
**Recommendation**: Regenerate from NAPI-RS auto-gen or extend the hand-written file.

---

## 4. NPM Package Quality

### 4.1 Test Coverage Summary

| Category | Packages | With Tests | Test Framework |
|----------|----------|-----------|----------------|
| NAPI native (npm/packages) | 8 | 0 | None |
| NAPI native (crate-side) | 10 | 2 | node:test (1), ava (1) |
| Pure TS/JS | ~30 | 6 | vitest (6), jest (3), node:test (12) |
| Platform binaries | 15 | 0 | N/A |

**Finding (CRITICAL -- NO JS-SIDE TESTS FOR NATIVE BINDINGS)**: Of the 8 NAPI native NPM packages, **zero** have JavaScript-side tests. Only `ruvector-gnn-node` (in its crate directory) has a `test/basic.test.js` with 15 tests, and `ruvector-node` has a `tests/basic.test.mjs` (ava, 15 tests). The other 8 NAPI crates have zero JS integration tests.

**Severity**: CRITICAL
**Impact**: No validation that the Rust-to-JS boundary works correctly at the JS level.

### 4.2 Test Framework Fragmentation (Confirmed from Phase 1)

Three test frameworks are in use across 56 packages:
- **node:test** (built-in): 15 packages (including some NAPI crate tests)
- **vitest**: 6 packages
- **jest**: 3 packages (`agentic-integration`, `burst-scaling`, `rvf`)

Plus **ava** in `ruvector-node/tests/basic.test.mjs` -- a fourth framework.

**Severity**: LOW (maintenance burden, not correctness)
**Recommendation**: Standardize on `vitest` or `node:test` for new packages.

### 4.3 package.json Quality

| Check | Pass | Fail | Details |
|-------|------|------|---------|
| `main` field present | 56/56 | 0 | All correct |
| `types` field present | ~45/56 | ~11 | Missing in `spiking-neural`, some platform pkgs |
| `engines.node` present | ~40/56 | ~16 | Missing in some packages |
| `exports` field | ~5/56 | ~51 | Most use legacy `main` only |
| Build script works | N/A | N/A | Not tested (requires native toolchain) |

**Finding (ENGINE VERSION INCONSISTENCY)**: Node.js engine requirements are inconsistent:
- `@ruvector/attention`: `>= 10` (napi9 requires Node 18+)
- `@ruvector/core`: `>= 18.0.0`
- `@ruvector/rvf-node`: `>= 16`
- `@ruvector/spiking-neural`: `>= 16.0.0`

Since all NAPI crates use `napi9` features, the minimum supported version is actually **Node.js 18**. The `>= 10` declaration in attention-node is misleading.

**Severity**: LOW
**Recommendation**: Set `engines.node` to `">= 18.0.0"` for all NAPI packages.

### 4.4 Duplicate Dependencies (Confirmed from Phase 1)

`@ruvector/agentic-integration` has:
- **Dual logging**: `winston` + `pino`
- **Dual HTTP frameworks**: `express` + `fastify`

**Severity**: MEDIUM
**Recommendation**: Consolidate to one of each. `pino` + `fastify` is the natural pair.

---

## 5. Cross-Platform Binary Distribution

### 5.1 Distribution Strategy

All NAPI-RS crates use the standard **optional dependencies** pattern:
1. Main package (e.g., `@ruvector/core`) has `optionalDependencies` pointing to platform-specific packages
2. Platform packages (e.g., `@ruvector/core-darwin-arm64`) contain the prebuilt `.node` binary
3. The `index.js` loader (auto-generated by NAPI-RS) detects platform/arch and loads the correct binary

### 5.2 Platform Coverage

| Package | linux-x64 | linux-arm64 | darwin-x64 | darwin-arm64 | win32-x64 | linux-musl | Fallback |
|---------|-----------|-------------|------------|--------------|-----------|-----------|----------|
| `@ruvector/core` | Yes | Yes | Yes | Yes | Yes | No | No |
| `@ruvector/graph-node` | Yes | Yes | Yes | Yes | Yes | No | No |
| `@ruvector/tiny-dancer` | Yes | Yes | Yes | Yes | Yes | No | No |
| `@ruvector/router` | Yes | Yes | Yes | Yes | Yes | No | No |
| `@ruvector/rvf-node` | Yes | Yes | Yes | Yes | Yes | No | No |
| `@ruvector/attention` | Yes | Yes | Yes | Yes | Yes | Partial* | No |
| `ruvector-gnn-node` | Yes | Yes | Yes | Yes | Yes | Yes | No |
| `ruvector-graph-transformer-node` | Yes | Yes | Yes | Yes | Yes | Yes | No |

*attention-node has musl package.json stubs but missing binaries.

### 5.3 Build-from-Source Fallback

**Finding (NO FALLBACK)**: None of the NAPI packages provide a `postinstall` build-from-source fallback. If the prebuilt binary is unavailable for a user's platform, `require()` will throw with a load error. There is no automatic `cargo build` fallback.

**Severity**: MEDIUM
**Recommendation**: Add a `postinstall` script that runs `napi build --release` if no prebuilt binary is found.

### 5.4 Misplaced Binaries (BUG)

**Finding (BUG -- WRONG BINARIES IN PLATFORM DIRECTORIES)**:

`ruvector-gnn-node/npm/`:
- `linux-arm64-gnu/` contains `ruvector-gnn.darwin-arm64.node` (WRONG)
- `linux-arm64-musl/` contains `ruvector-gnn.darwin-arm64.node` and `ruvector-gnn.linux-x64-gnu.node` (WRONG)
- `linux-x64-musl/` contains `ruvector-gnn.darwin-arm64.node` and `ruvector-gnn.linux-x64-gnu.node` (WRONG)
- `win32-x64-msvc/` contains `ruvector-gnn.darwin-arm64.node` (WRONG)

`ruvector-graph-transformer-node/npm/`:
- `darwin-arm64/` contains `ruvector-graph-transformer.darwin-x64.node` (WRONG)
- `linux-x64-gnu/` contains `ruvector-graph-transformer.darwin-x64.node` (WRONG)
- `linux-x64-musl/` contains `ruvector-graph-transformer.darwin-x64.node` and `linux-x64-gnu.node` (WRONG extra)

These extra files are harmless if the `index.js` loader selects by filename convention, but they waste ~5MB of space in the published package and could confuse tooling.

**Severity**: LOW (non-functional, but wastes package size)
**Recommendation**: Clean up platform directories to contain only the correctly named binary.

### 5.5 Node.js Version Support

All NAPI crates target `napi9` which requires **Node.js 18.17.0+** (the LTS version that shipped with N-API v9). The CI workflow in `ruvector-gnn-node` uses Node 18 for testing.

---

## 6. Test Analysis

### 6.1 JavaScript-Side Tests

| Crate/Package | Test File | Framework | Test Count | Covers |
|--------------|-----------|-----------|-----------|--------|
| `ruvector-gnn-node` | `test/basic.test.js` | node:test | 15 | Layer creation, forward pass, serialization, compression round-trip, search, error cases |
| `ruvector-node` | `tests/basic.test.mjs` | ava | 15 | VectorDB CRUD, search, batch ops, HNSW config, concurrent ops, stress test (1000 vectors) |
| All other NAPI crates (8) | None | N/A | 0 | Nothing |

### 6.2 Rust-Side Tests (in binding crates)

| Crate | #[test] Count | Covers |
|-------|-------------|--------|
| `agentic-robotics-node` | 5 | Node creation, publisher/subscriber creation, publish, list |
| `ruvector-graph-node` (transactions.rs) | 2 | Transaction lifecycle, rollback |
| `ruvector-graph-transformer-node` (transformer.rs) | 20 | Proof gates, sublinear attention, Hamiltonian, spiking, Hebbian, verified training, manifold, causal, game theory |
| All other crates | 0 | Nothing |

### 6.3 Memory Leak Tests

**Finding (NO MEMORY LEAK TESTS)**: Zero packages have dedicated memory leak tests. The `ruvector-node` test file has a "memory stress test" that inserts 1000 vectors, but it only checks correctness -- it does not monitor RSS/heap growth or use `FinalizationRegistry`/`WeakRef` to verify GC behavior.

No packages use:
- `v8.getHeapStatistics()` for heap monitoring
- `process.memoryUsage()` for RSS tracking
- `FinalizationRegistry` to verify Rust object cleanup
- Repeated alloc/dealloc cycles to detect leaks

**Severity**: HIGH
**Impact**: Memory leaks in the Rust-JS boundary would go undetected. Given that NAPI-RS handles GC integration, leaks are unlikely but unverified.

### 6.4 Integration Test Gap

No tests exercise the **full stack** of:
1. JS creates Rust object via NAPI
2. Performs operations
3. Verifies Rust state through JS API
4. Destroys object and verifies cleanup

The `ruvector-gnn-node` and `ruvector-node` tests come closest but lack cleanup verification.

---

## 7. Summary of Findings

### Critical (3)

| ID | Finding | Location | Recommendation |
|----|---------|----------|----------------|
| D7-C1 | No JS-side tests for 8 of 10 NAPI binding packages | All NAPI npm packages | Add at minimum smoke tests exercising constructor + key method + error path for each |
| D7-C2 | Panic-on-poisoned-lock cascade risk | `ruvector-graph-node` (15), `ruvector-mincut-node` (8), `ruvector-node` (14) | Replace `expect("RwLock poisoned")` and `.unwrap()` with `.map_err()` returning JS Error |
| D7-C3 | No memory leak tests for any native binding | All NAPI packages | Add GC verification tests using `FinalizationRegistry` + repeated alloc/dealloc cycles |

### High (2)

| ID | Finding | Location | Recommendation |
|----|---------|----------|----------------|
| D7-H1 | `ruvector-graph-transformer-node` returns `any` for 14 of its 18 methods | `index.d.ts` (14 occurrences) | Define specific TypeScript interfaces for each return type |
| D7-H2 | No build-from-source fallback for any NAPI package | All NAPI packages | Add `postinstall` script with `napi build` fallback |

### Medium (5)

| ID | Finding | Location | Recommendation |
|----|---------|----------|----------------|
| D7-M1 | `@ruvector/core` index.d.ts incomplete (misses CollectionManager, metrics, health, filters) | `npm/packages/core/index.d.ts` | Regenerate or extend to cover full API |
| D7-M2 | `spiking-neural` has no TypeScript definitions | `npm/packages/spiking-neural/` | Add `index.d.ts` |
| D7-M3 | No explicit `close()` / `dispose()` for `GraphDatabase` and `VectorDB` | `ruvector-graph-node`, `ruvector-node` | Add deterministic resource cleanup methods |
| D7-M4 | NAPI version inconsistency (some crates override workspace) | `ruvector-attention-node`, `ruvector-graph-transformer-node` | Use `napi = { workspace = true }` everywhere |
| D7-M5 | Duplicate logging (winston+pino) and HTTP frameworks (express+fastify) in agentic-integration | `npm/packages/agentic-integration/` | Consolidate to pino + fastify |

### Low (4)

| ID | Finding | Location | Recommendation |
|----|---------|----------|----------------|
| D7-L1 | Misplaced platform binaries (darwin binaries in linux/win dirs) | `ruvector-gnn-node/npm/`, `ruvector-graph-transformer-node/npm/` | Clean platform directories |
| D7-L2 | `engines.node` claims `>= 10` but napi9 requires `>= 18` | `ruvector-attention-node/package.json` | Set to `>= 18.0.0` |
| D7-L3 | Four test frameworks (vitest, jest, node:test, ava) across packages | Various | Standardize on vitest or node:test |
| D7-L4 | Transaction manager in graph-node is purely in-memory (no actual ACID) | `ruvector-graph-node/src/transactions.rs` | Document limitation or implement real tx isolation |

---

## 8. Positive Findings

1. **Zero unsafe in NAPI bindings**: 9 of 10 NAPI crates have zero `unsafe` blocks. The one exception is a justified WASM ABI stub.
2. **Consistent error propagation**: All crates convert Rust errors to proper JS Error objects. No silent failures.
3. **Good async patterns**: Crates correctly use `tokio::task::spawn_blocking` for CPU-bound work, avoiding event loop blocking.
4. **Comprehensive attention bindings**: `ruvector-attention-node` (2,515 LOC) covers 7 attention variants, 5 training utilities, async batch processing, and graph attention -- all with type-safe NAPI bindings.
5. **Well-structured GNN tests**: The `ruvector-gnn-node/test/basic.test.js` covers 15 scenarios including edge cases (empty input, invalid parameters) and round-trip verification.
6. **5-platform binary coverage**: All NAPI packages ship prebuilt binaries for linux-x64, linux-arm64, darwin-x64, darwin-arm64, and win32-x64.
7. **Graph transformer self-contained**: `ruvector-graph-transformer-node` embeds its own implementation (1,356 LOC in `transformer.rs`) to avoid coupling with the evolving core crate -- a pragmatic architectural choice with 20 Rust unit tests.

---

## 9. Risk Matrix

| Risk | Probability | Impact | Finding |
|------|------------|--------|---------|
| Process crash from poisoned lock | Medium | Critical | D7-C2 |
| Runtime failure on untested platform | Medium | High | D7-C1, D7-H2 |
| Memory leak undetected | Low | High | D7-C3 |
| Type errors in graph-transformer consumers | High | Medium | D7-H1 |
| Data loss on abrupt exit (no close()) | Medium | Medium | D7-M3 |
| Wrong binary loaded (misplaced files) | Very Low | High | D7-L1 |

---

## 10. Recommended Priority Order for Remediation

1. **D7-C2**: Fix poisoned lock panic risk (replace unwrap/expect with map_err)
2. **D7-C1**: Add JS smoke tests for all 8 untested NAPI packages
3. **D7-H1**: Add TypeScript interfaces for graph-transformer-node return types
4. **D7-M3**: Add explicit close()/dispose() to GraphDatabase and VectorDB
5. **D7-C3**: Add memory leak regression tests
6. **D7-M1**: Complete core package TypeScript definitions
7. **D7-H2**: Add postinstall build-from-source fallback
8. **D7-L1**: Clean up misplaced platform binaries
