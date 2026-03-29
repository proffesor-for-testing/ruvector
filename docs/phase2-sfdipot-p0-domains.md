# Phase 2: SFDIPOT Product Factors Analysis -- P0 Critical Domains

**Framework**: James Bach's Heuristic Test Strategy Model (HTSM) -- Product Factors (SFDIPOT)
**Date**: 2026-03-29
**Scope**: Three P0 critical domains of the RuVector monorepo
**Method**: Code-level analysis of Rust source, Cargo.toml manifests, tests, and benchmarks

---

## Domain Map

| Domain | Crates | LOC (approx) | Complexity |
|--------|--------|--------------|------------|
| **D1: Core Vector DB** | ruvector-core, ruvector-collections, ruvector-filter, ruvector-math, ruvector-metrics | ~12K | High (SIMD, quantization, HNSW) |
| **D2: Graph Database** | ruvector-graph, ruvector-graph-node, ruvector-graph-wasm, ruvector-graph-transformer | ~15K | Very High (Cypher parser, MVCC, distributed, hypergraphs) |
| **D3: Distributed Systems** | ruvector-raft, ruvector-replication, ruvector-cluster, ruvector-delta-* | ~10K | Critical (consensus, CRDTs, failover, sharding) |

---

## D1: Core Vector DB

### S -- Structure

**Current State**:
- **Module architecture**: ruvector-core is the foundational crate. It exposes `VectorDB` as the main entry point, backed by a `VectorIndex` trait with two implementations: `FlatIndex` (brute-force) and `HnswIndex` (approximate nearest neighbor via hnsw_rs).
- **Storage layer**: `VectorStorage` wraps redb (embedded key-value store) with a global `DB_POOL` (Lazy<Mutex<HashMap>>) for connection sharing. A `storage_memory` module provides a WASM fallback.
- **Feature flags**: 8 feature flags (`simd`, `parallel`, `storage`, `hnsw`, `memory-only`, `api-embeddings`, `onnx-embeddings`, `real-embeddings`) control conditional compilation. Default includes `simd`, `storage`, `hnsw`, `api-embeddings`, `parallel`.
- **Dependencies**: Core depends on redb, simsimd, hnsw_rs, rayon, crossbeam, rkyv, bincode, serde, ndarray, dashmap, parking_lot.
- **Advanced features**: Modules for quantization (scalar, int4, product, binary), arena allocation, lock-free data structures, cache-optimized SoA storage, advanced features (conformal prediction, hybrid search, MMR, sparse vectors, DiskANN, graph-RAG).
- **Layer boundaries**: Types (types.rs) -> Distance (distance.rs) -> Index (index/) -> VectorDB (vector_db.rs) -> Storage (storage.rs). Clean separation.

**Risks**:
- **Global static DB_POOL**: `Lazy<Mutex<HashMap<PathBuf, Arc<Database>>>>` is a global mutable singleton. Under heavy concurrent open/close, the Mutex could become a bottleneck or leak entries if `VectorStorage` instances are not properly dropped.
- **HNSW deletion is a stub**: `HnswIndex::remove()` removes from internal maps but NOT from the hnsw_rs graph structure. Comment at line 339-341 says "hnsw_rs doesn't support direct deletion ... This is a known limitation." Deleted vectors can still appear in search results as ghost entries.
- **Deserialization rebuilds entire index**: `HnswIndex::deserialize()` iterates all vectors with O(n * log(n)) insert cost. For 10M+ vectors, this could take minutes on restart.
- **Feature flag combinatorics**: 8 flags produce 256 configurations. Not all are tested. The WASM build path (`not(feature = "storage")`) has different storage semantics.

**Test Ideas**:
1. Insert 1M vectors, delete 50% of them, then search -- confirm deleted vectors never appear in results. (Targets HNSW ghost entry bug.) **P0**
2. Open 100 concurrent `VectorStorage` instances pointing to the same database path, perform interleaved writes -- measure whether `DB_POOL` Mutex causes contention or deadlock. **P1**
3. Build ruvector-core with `memory-only` feature, insert 10K vectors, serialize to bytes, deserialize, search -- confirm round-trip correctness without redb. **P1**
4. Build with every non-default feature combination (`--no-default-features --features "simd"`, `--features "storage,hnsw"`, etc.) -- confirm compilation succeeds. **P2**
5. Trigger `VectorDB::new()` with a corrupted redb file -- confirm graceful error, not panic. **P1**

---

### F -- Function

**Current State**:
- **Core operations**: insert (single + batch), search (KNN via HNSW or flat), delete, get-by-id, list-keys.
- **Distance metrics**: Euclidean, Cosine, DotProduct, Manhattan. SimSIMD used on native, scalar fallback on WASM.
- **Quantization**: Scalar (int8, 4x compression), Int4 (8x), Product Quantization (8-16x), Binary (32x). Each has its own distance computation path.
- **Search enrichment**: After HNSW returns IDs+scores, `VectorDB::search()` fetches full vectors and metadata from storage, then applies metadata filters.
- **Post-restart recovery**: `VectorDB::new()` detects persisted vectors and rebuilds the HNSW index from storage. Config is persisted to `CONFIG_TABLE`.
- **Embeddings**: `HashEmbedding` (placeholder, character-based -- explicitly warned as NOT semantic), `OnnxEmbedding` (real ONNX runtime), `ApiEmbedding` (HTTP-based). The code has a compile-time deprecation warning about AgenticDB's placeholder embeddings.

**Risks**:
- **Metadata filter is post-hoc**: Filters are applied AFTER HNSW returns K results. If K=10 and 8 fail the filter, only 2 results are returned. No over-fetching logic.
- **Batch insert non-atomicity**: `insert_batch` writes to storage in one redb transaction, then adds to HNSW index separately. If the process crashes between storage commit and HNSW insertion, the data is persisted but not indexed. On restart, the index is rebuilt from storage, so eventual consistency is maintained -- but during the session, the vectors are invisible to search.
- **Search on empty DB does not short-circuit**: No fast path for zero-vector databases.
- **`euclidean_distance` uses `.expect()` on SimSIMD**: Line 31 -- `simsimd::SpatialSimilarity::sqeuclidean(a, b).expect("SimSIMD euclidean failed")`. If SimSIMD has an internal error (e.g., unsupported dimension), this panics.

**Test Ideas**:
1. Insert 1000 vectors with metadata `{category: "A"}`, search with filter `{category: "B"}` -- confirm empty result set, not panic or partial results. **P1**
2. Insert a batch of 10K vectors, kill the process mid-insert (simulate with `std::process::abort()` after storage commit but before index build), restart -- confirm all 10K are searchable after restart. **P0**
3. Provide a 0-dimensional vector to `insert()` -- confirm `DimensionMismatch` error, not panic. **P1**
4. Search with k=100 on a database with 5 vectors -- confirm exactly 5 results returned, not an error. **P2**
5. Create a VectorDB with Cosine metric, insert the zero vector `[0.0, 0.0, ..., 0.0]`, search for it -- confirm no division-by-zero panic (cosine_distance checks `denom > 1e-8`). **P0**

---

### D -- Data

**Current State**:
- **Central data types**: `VectorEntry` (id: Option<String>, vector: Vec<f32>, metadata: Option<HashMap<String, serde_json::Value>>). `SearchResult` (id, score, vector, metadata). `SearchQuery` (vector, k, filter, ef_search).
- **Serialization**: Vectors stored as bincode-encoded `Vec<f32>` in redb. Metadata stored as JSON strings. Config stored as JSON. HNSW state uses bincode with `Encode`/`Decode` derives.
- **Dimension limits**: Configurable via `DbOptions.dimensions` (default 384). No hard upper limit enforced -- only mismatch checks.
- **Vector ID**: String type (`type VectorId = String`). Auto-generated as UUID v4 if not provided.
- **Validation**: Dimension mismatch checked at insert and search. No NaN/Inf validation. No metadata size limits.

**Risks**:
- **NaN/Inf vectors**: Inserting a vector containing NaN or Infinity values is not validated. SimSIMD and HNSW behavior with NaN is undefined. Could corrupt the index graph.
- **Unbounded metadata**: Metadata is `HashMap<String, serde_json::Value>` with no size limit. A single vector entry with a 1GB JSON metadata blob would be persisted.
- **Type confusion on deserialization**: If a redb database is opened with wrong dimensions, `load_config()` returns the stored config and overrides the user's options. But if the config is missing (old database), the user's dimensions are used, potentially causing mismatches with stored vectors.
- **String IDs have no uniqueness enforcement at the storage level**: Inserting with the same ID overwrites silently. No upsert semantics documented.

**Test Ideas**:
1. Insert a vector containing `f32::NAN` and `f32::INFINITY` -- confirm graceful rejection or at minimum, that subsequent searches do not panic or return corrupted results. **P0**
2. Insert 1000 vectors with randomly generated IDs, then insert with a duplicate ID -- confirm the second insert overwrites the first (or returns an error, depending on intended behavior). **P1**
3. Create a database with dimensions=384, close it, reopen with dimensions=768 in the options -- confirm the stored config (384) takes precedence and operations succeed. **P1**
4. Insert a vector with metadata containing deeply nested JSON (100 levels) -- confirm serialization/deserialization round-trip is correct. **P2**
5. Open a database with a truncated/corrupted redb file -- confirm `RuvectorError::DatabaseError`, not a panic. **P1**

---

### I -- Interfaces

**Current State**:
- **Rust API**: `VectorDB` struct with `new()`, `insert()`, `insert_batch()`, `search()`, `delete()`, `get()`, `len()`, `keys()`. Clean trait-based index abstraction via `VectorIndex`.
- **Error types**: `RuvectorError` enum with 14 variants, all string-based messages except `DimensionMismatch` (structured) and `IoError` (from std::io). Implements `From` for 5 redb error types.
- **Feature-gated modules**: `agenticdb`, `lockfree`, `storage` are conditionally compiled. The public API surface changes based on features.
- **NAPI binding**: Via ruvector-graph-node crate (see D2).
- **WASM binding**: Via separate crates.
- **No HTTP/REST API in core**: The core is a library crate only.

**Risks**:
- **Error type is not `Clone` or `PartialEq`**: `RuvectorError` derives only `Error, Debug`. Testing error conditions requires matching on string messages, which is brittle.
- **`VectorIndex` trait requires `&mut self` for `add()`**: This forces callers to hold a write lock. The `VectorDB` uses `Arc<RwLock<Box<dyn VectorIndex>>>`, meaning all inserts are serialized even if the underlying index supports concurrent writes.
- **No versioning in serialization format**: bincode-encoded HNSW state has no version field. Format changes would break deserialization of existing databases.

**Test Ideas**:
1. Attempt to use `VectorDB` from 10 concurrent threads (5 readers, 5 writers) -- measure throughput and confirm no deadlocks. **P0**
2. Serialize an HNSW index with version N of the code, then attempt to deserialize with a modified schema (added field) -- confirm graceful error, not undefined behavior. **P1**
3. Call every public method on `VectorDB` with edge-case inputs (empty string ID, k=0, empty vector, k=usize::MAX) -- confirm all return `Result::Err`, not panic. **P1**
4. Build ruvector-core as `cdylib` for FFI -- confirm the public API is consumable from C. **P2**
5. Confirm that `RuvectorError` Display messages are unique enough to distinguish all 14 variants programmatically. **P2**

---

### P -- Platform

**Current State**:
- **Targets**: Native x86_64 (Linux, macOS, Windows), ARM64/aarch64 (Apple Silicon, Linux ARM), WASM32 (browser, Node.js).
- **SIMD dispatch**: Runtime detection on x86_64 (`is_x86_feature_detected!("avx512f")`, `"avx2"`, `"fma"`). Compile-time on aarch64 (NEON). Scalar fallback on WASM and unknown architectures.
- **SIMD coverage**: Custom intrinsics in `simd_intrinsics.rs` for Euclidean, Cosine, DotProduct, Manhattan. AVX-512, AVX2+FMA, AVX2, NEON (with 4x unrolled variant for vectors >= 64 elements), scalar fallback.
- **Storage**: redb (native only), in-memory HashMap (WASM).
- **Parallelism**: Rayon for batch operations (native only). Sequential fallback on WASM.
- **Rust version**: Workspace-level MSRV (not explicitly checked in this file, but `rust-version.workspace = true`).

**Risks**:
- **No CI for WASM build**: The `memory-only` feature path is likely less tested. WASM builds exclude `storage`, `hnsw`, `parallel`, `simd`, `api-embeddings`.
- **AVX-512 path has narrow testing**: AVX-512 is only available on specific Intel CPUs (Skylake-X and newer). CI runners likely use older hardware.
- **ARM NEON path uses `#[inline(always)]` + `unsafe`**: Assembly-level bugs in NEON intrinsics are hard to detect without ARM CI.
- **No `#[target_feature]` guards on some NEON functions**: The code uses `#[cfg(target_arch = "aarch64")]` but some NEON intrinsics require specific feature gates.

**Test Ideas**:
1. Run the full test suite under `cargo test --target wasm32-unknown-unknown --no-default-features --features "memory-only"` -- confirm all tests pass. **P0**
2. Run SIMD correctness tests (existing `test_simd_correctness.rs`) on x86_64 with AVX2 and on aarch64 with NEON -- compare distance results with scalar baseline within epsilon tolerance. **P0**
3. Run benchmarks on vectors of dimensions 1, 3, 7 (non-aligned to SIMD widths) -- confirm SIMD tail-handling is correct. **P1**
4. Build and run on a machine without AVX2 (e.g., older VM) -- confirm runtime detection falls back to scalar. **P2**
5. Compile with `RUSTFLAGS="-C target-cpu=native"` on Apple M4 -- confirm NEON unrolled path is used for 384-dim vectors. **P2**

---

### O -- Operations

**Current State**:
- **Metrics**: ruvector-metrics crate provides Prometheus-compatible metrics: search/insert/delete counters and latency histograms, vector counts, memory usage, uptime. Global lazy_static registries.
- **Health checks**: `HealthChecker` in ruvector-metrics provides health/readiness endpoints.
- **Configuration**: `DbOptions` struct with dimensions, distance_metric, storage_path, hnsw_config, quantization. Persisted to redb CONFIG_TABLE.
- **Lock-free stats**: `LockFreeStats` in lockfree.rs tracks queries, inserts, deletes, and average latency with atomic counters.
- **No built-in backup/restore**: redb files can be copied, but there is no snapshot API.
- **No graceful shutdown**: Dropping `VectorDB` drops the storage Arc. The global `DB_POOL` retains the redb Database Arc indefinitely (memory leak if many databases are opened and closed).

**Risks**:
- **DB_POOL memory leak**: The global pool `Lazy<Mutex<HashMap<PathBuf, Arc<Database>>>>` never evicts entries. Opening 1000 databases over a long-running process leaks 1000 Arc<Database> references.
- **No WAL or fsync guarantees documented**: redb provides ACID guarantees, but the interaction with HNSW in-memory index means crash recovery depends on the redb commit being durable. If redb uses memory-mapped writes without explicit fsync, data loss is possible on power failure.
- **Prometheus metrics use global registries**: Multiple test processes or instances in the same process will collide on metric names.

**Test Ideas**:
1. Open 100 databases with different paths, close all handles, check `DB_POOL` size -- confirm entries are leaked (documenting current behavior) or cleaned up. **P1**
2. Insert 10K vectors, force-kill the process (SIGKILL), restart -- confirm data integrity (no partial writes in redb). **P0**
3. Record 10K search operations, export Prometheus metrics -- confirm counters and histograms reflect accurate counts and latency percentiles. **P2**
4. Trigger `HealthChecker` when the storage directory is on a read-only filesystem -- confirm health status reflects degraded state. **P2**
5. Run a 24-hour soak test inserting and searching continuously -- monitor memory growth for leaks in DB_POOL, DashMap, or HNSW index. **P1**

---

### T -- Time

**Current State**:
- **Concurrent access**: `VectorDB` uses `Arc<RwLock<Box<dyn VectorIndex>>>`. Reads (search) take a read lock; writes (insert, delete) take a write lock. `VectorStorage` uses redb transactions (serialized writes).
- **Lock-free structures**: `LockFreeCounter`, `LockFreeStats`, `AtomicVectorPool`, `LockFreeBatchProcessor`, `ObjectPool` -- all in lockfree.rs, gated behind `parallel` feature.
- **Batch processing**: `LockFreeBatchProcessor` uses `ArrayQueue` for work distribution and `SegQueue` for results. Includes `is_done()` check based on atomic counters.
- **No TTL**: Vectors do not expire. No time-based eviction.
- **Startup time**: Index rebuilding from storage is synchronous and blocking. For large databases, this could take minutes.

**Risks**:
- **Write starvation**: Under heavy read load (many concurrent searches), writes may be starved because `RwLock` on the HNSW index is reader-biased in parking_lot's default fairness mode.
- **ObjectPool spin-wait**: When the pool is at capacity, `ObjectPool::acquire()` enters a `loop { spin_loop() }` -- an unbounded busy-wait. Under contention, this wastes CPU indefinitely.
- **Batch processor `is_done()` is racy**: `pending()` and `completed()` read two separate atomics. Between the reads, new items could be submitted, making `is_done()` return false positives.
- **No timeout on redb transactions**: A long-running redb write transaction blocks all other writes indefinitely.

**Test Ideas**:
1. Spawn 50 reader threads and 5 writer threads, run for 60 seconds -- measure p99 write latency to detect write starvation. **P0**
2. Create an `ObjectPool` with capacity=1, then spawn 100 threads all calling `acquire()` simultaneously -- confirm no thread hangs indefinitely (or document the spin-wait as a known limitation). **P1**
3. Submit 1000 items to `LockFreeBatchProcessor`, process 500, check `is_done()` from a concurrent thread -- confirm it returns false. **P2**
4. Measure `VectorDB::new()` startup time with 1M persisted vectors -- confirm it completes within 60 seconds. **P1**
5. Insert vectors continuously while searching concurrently -- measure whether search latency degrades under sustained write load. **P1**

---

## D2: Graph Database

### S -- Structure

**Current State**:
- **Core architecture**: `GraphDB` struct holds `DashMap<NodeId, Node>`, `DashMap<EdgeId, Edge>`, `DashMap<HyperedgeId, Hyperedge>` plus 5 index structures (LabelIndex, PropertyIndex, EdgeTypeIndex, AdjacencyIndex, HyperedgeNodeIndex).
- **Cypher query pipeline**: Lexer (nom-based tokenizer) -> Parser (recursive descent) -> AST -> Semantic analysis -> Optimizer -> Executor (pipeline-based with cache, parallel execution, stats).
- **Transaction system**: MVCC-based `TransactionManager` with `Version<T>` history, per-entity version chains stored in `DashMap<K, Vec<Version<T>>>`.
- **Distributed layer**: Behind `distributed` feature flag. Includes `Coordinator`, `GraphShard`, `Federation`, `GossipMembership`, `GraphReplication`, `RpcClient`/`RpcServer`. Depends on ruvector-raft, ruvector-cluster, ruvector-replication.
- **Hybrid vector-graph**: `hybrid/` module with `SemanticSearch`, `GraphNeuralEngine`, `RagEngine`, `VectorCypherParser`, `HybridIndex`.
- **Optimization layer**: Adaptive radix trees, bloom filters, cache hierarchy, index compression, memory pool, query JIT (stub), SIMD traversal.
- **External bindings**: ruvector-graph-node (NAPI-RS for Node.js), ruvector-graph-wasm (wasm-bindgen for browsers).
- **Feature flags**: 14 features including `full`, `simd`, `storage`, `async-runtime`, `compression`, `distributed`, `federation`, `metrics`, `wasm`, etc.

**Risks**:
- **DashMap everywhere**: All 5 indexes are separate DashMaps with no cross-index consistency guarantees. Creating a node updates `nodes`, `label_index`, and `property_index` in sequence -- a crash between `label_index.add_node()` and `nodes.insert()` could leave indexes inconsistent.
- **MVCC version chains grow unbounded**: `TransactionManager` never garbage-collects old versions. Long-running workloads will accumulate O(updates) version entries per entity.
- **GraphDB and TransactionManager are separate**: `GraphDB` operations (create_node, create_edge) do NOT participate in transactions. The `TransactionManager` has its own separate storage. These are two parallel systems that don't interoperate.
- **Placeholder test**: `lib.rs` has `test_placeholder` that just asserts `true`. This suggests test coverage may be thin for the core module itself.
- **ruvector-graph-wasm duplicates GraphDB**: The WASM crate defines its OWN `GraphDB` struct with its own `HashMap`-based storage, completely separate from `ruvector_graph::GraphDB`. Two implementations to maintain.

**Test Ideas**:
1. Create 10K nodes and 50K edges, then crash-simulate during a bulk create_node loop -- restart and verify all indexes (label, property, adjacency) are consistent with the node DashMap. **P0**
2. Run 1000 transactions against TransactionManager, each writing 10 nodes, then verify MVCC version chain memory consumption -- confirm it does not grow without bound. **P1**
3. Create a node via `GraphDB::create_node()` and separately via `TransactionManager::begin() + write_node() + commit()` -- confirm both paths produce queryable results (or document they are independent). **P0**
4. Execute a complex Cypher query (`MATCH (a)-[:KNOWS]->(b)-[:WORKS_AT]->(c) WHERE c.name = 'Acme' RETURN a, b`) -- confirm parser, optimizer, and executor produce correct results. **P1**
5. Run the same graph operations through ruvector-graph (native), ruvector-graph-node (NAPI), and ruvector-graph-wasm (WASM) -- confirm identical results. **P1**

---

### F -- Function

**Current State**:
- **Node operations**: create, get, delete, get_by_label, get_by_property. Node has labels (Vec<Label>) and properties (HashMap<String, PropertyValue>).
- **Edge operations**: create (validates source/target exist), get, delete, get_by_type, get_outgoing, get_incoming. Edge has from, to, relation_type, properties.
- **Hyperedge operations**: create (validates all nodes exist), get, get_by_node. Hyperedge connects N nodes with description, weight/confidence.
- **Cypher support**: MATCH, CREATE, RETURN, WHERE, ORDER BY, LIMIT, DELETE, SET, MERGE, WITH, UNION, OPTIONAL MATCH, EXISTS, CASE. Recursive descent parser with semantic analysis.
- **Transaction support**: BEGIN, COMMIT, ROLLBACK with 4 isolation levels (ReadUncommitted, ReadCommitted, RepeatableRead, Serializable). MVCC visibility based on start timestamp and transaction IDs.
- **Distributed queries**: Query planning, shard routing, result aggregation, cross-shard joins.

**Risks**:
- **Node deletion does not cascade to edges**: `GraphDB::delete_node()` removes the node and updates label/property indexes but does NOT delete associated edges. Orphaned edges pointing to deleted nodes remain.
- **Edge creation validates node existence but does not hold locks**: Between the `contains_key()` check and the edge insert, a concurrent delete could remove the source/target node. No transactional guarantee.
- **MVCC read excludes own transaction's writes from version store**: Line 224 in transaction.rs: `v.created_by != txn_id` means a transaction cannot see its own committed writes via the MVCC store. The write-set check handles uncommitted writes, but after commit, the filter would exclude them for the same txn_id.
- **Cypher parser does not handle all edge cases**: The parser is a hand-written recursive descent. Complex nested patterns, variable-length paths, and aggregation functions may have gaps.
- **WASM Cypher is a simplified stub**: `ruvector-graph-wasm` has `execute_match_query` returning empty results. The WASM binding does NOT use the ruvector-graph Cypher parser.

**Test Ideas**:
1. Create nodes A and B, create edge A->B, delete node A, query edge -- confirm the edge still exists (documenting current behavior) or is cascade-deleted. **P0**
2. Spawn 2 threads: Thread 1 creates node N, Thread 2 concurrently deletes node N while Thread 1 creates edge to N -- confirm no dangling edge or panic. **P0**
3. Begin a transaction, write node X, commit, begin a new transaction, read node X -- confirm visibility is correct across transaction boundaries. **P1**
4. Send 100 different Cypher queries (including malformed ones, SQL injection attempts, deeply nested patterns) to the parser -- confirm graceful error handling for every invalid query. **P1**
5. Execute `MATCH (n) RETURN count(n)` on a graph with 100K nodes -- measure query execution time and confirm aggregation correctness. **P2**

---

### D -- Data

**Current State**:
- **Node data**: `NodeId` (String), `Label` (name: String), `Properties` (HashMap<String, PropertyValue>). `PropertyValue` supports String, Integer (i64), Float (f64), Boolean, List, Map, Null.
- **Edge data**: `EdgeId` (String), from/to NodeId, `RelationType` (String), Properties.
- **Hyperedge data**: Vec<NodeId>, description (String), embedding (Vec<f32>), confidence (f32), metadata.
- **Serialization**: Nodes, Edges use bincode (Encode/Decode derives) for storage, serde for JSON. Properties use serde_json::Value at the NAPI boundary.
- **Graph storage**: `GraphStorage` (redb-based) with separate tables for nodes, edges, hyperedges, plus their ID lists.
- **Cypher AST**: Rich type hierarchy -- Query, Statement, MatchClause, Pattern (Node, Relationship), Expression (Literal, Variable, BinaryOp, FunctionCall, etc.).

**Risks**:
- **Property value type coercion at NAPI boundary**: ruvector-graph-node converts JavaScript values to `PropertyValue`. Type mismatches (JS number -> i64 vs f64, BigInt overflow, nested objects) are potential data corruption vectors.
- **No schema enforcement**: Nodes with label "Person" can have completely different property sets. No way to enforce that all "Person" nodes have a "name" property.
- **Embedding dimension mismatch in hyperedges**: Hyperedge embeddings have no dimension validation against an expected value. Mixing 384-dim and 768-dim embeddings in the same graph is silently accepted.
- **Cypher literal parsing**: The lexer/parser handles string literals, integers, floats, booleans. Special characters in string literals (Unicode, escape sequences, null bytes) may not be fully handled.

**Test Ideas**:
1. Create nodes via NAPI with JavaScript BigInt values, negative numbers, floating-point edge cases (Infinity, NaN, -0) -- confirm PropertyValue handles them correctly or rejects gracefully. **P1**
2. Create 100 hyperedges with embeddings of different dimensions (128, 256, 384) in the same graph -- search by embedding and confirm results are meaningful or an error is raised. **P1**
3. Store a node with a property value containing 1MB of text -- persist to GraphStorage, reload -- confirm round-trip integrity. **P2**
4. Parse a Cypher query with Unicode identifiers (Chinese characters, emoji, zero-width joiners) -- confirm parser handles or rejects gracefully. **P2**
5. Create a graph with 1M nodes, persist to redb, reload via `GraphDB::with_storage()` -- measure load time and confirm all indexes are correctly rebuilt. **P1**

---

### I -- Interfaces

**Current State**:
- **Rust API**: `GraphDB` (in-memory or with storage), `TransactionManager`, `parse_cypher()`, query executor pipeline.
- **NAPI (Node.js)**: `GraphDatabase` class with `createNode()`, `createEdge()`, `createHyperedge()`, `query()`, `querySync()`, `searchHyperedges()`, `kHopNeighbors()`, `begin()`, `commit()`, `rollback()`, `batchInsert()`, `subscribe()`, `stats()`. All async methods use `tokio::task::spawn_blocking`.
- **WASM**: `GraphDB` class with `createNode()`, `createEdge()`, `createHyperedge()`, `query()`, `getNode()`, `getEdge()`, `deleteNode()`, `deleteEdge()`, `importCypher()`, `exportCypher()`, `stats()`. Uses parking_lot::Mutex (not async).
- **Error handling**: `GraphError` with 20 variants. NAPI converts to `napi::Error` via `Error::from_reason()`. WASM converts to `JsValue::from_str()`.

**Risks**:
- **NAPI uses `std::sync::RwLock` instead of `parking_lot::RwLock`**: The NAPI bindings at line 37 use `std::sync::RwLock` which can poison on panic. A panic in `spawn_blocking` would permanently poison the lock, making subsequent operations fail with "RwLock poisoned".
- **WASM uses `parking_lot::Mutex` (blocking)**: In a single-threaded WASM environment, `Mutex::lock()` on an already-locked mutex will deadlock. The WASM bindings are NOT safe for re-entrant calls.
- **subscribe() is a stub**: Line 521 in graph-node/src/lib.rs -- `pub fn subscribe(&self, callback: JsFunction) -> Result<()> { Ok(()) }`. The callback is never invoked. Users expecting change notifications will get silent no-ops.
- **Two separate transaction systems**: NAPI exposes `transactions::TransactionManager` (its own implementation) while the Rust core has `ruvector_graph::TransactionManager`. These are unrelated.

**Test Ideas**:
1. From Node.js, call `createNode()` then immediately call `query('MATCH (n) RETURN n')` -- confirm the node is visible in query results (tests NAPI async ordering). **P0**
2. From WASM, call `createNode()` from within a `query()` callback (re-entrant call) -- confirm deadlock is handled or prevented. **P1**
3. Trigger a panic inside `spawn_blocking` (e.g., via a poisoned RwLock), then call another NAPI method -- confirm the error message indicates RwLock poisoning, not a segfault. **P1**
4. Call `subscribe()` from Node.js, create nodes, verify whether the callback is invoked -- document that it is a no-op. **P2**
5. Serialize a graph to Cypher via `exportCypher()`, then `importCypher()` on a new WASM instance -- confirm round-trip fidelity. **P1**

---

### P -- Platform

**Current State**:
- **Native**: Full feature set including SIMD, storage (redb), async (tokio), distributed, compression (zstd, lz4).
- **Node.js**: NAPI-RS bindings. Pre-built binaries for linux-x64, darwin-arm64, win32-x64. Requires tokio runtime.
- **WASM**: wasm-bindgen. Uses `console_error_panic_hook` and `tracing_wasm`. No persistence (in-memory only). No async runtime (parking_lot Mutex, not tokio).
- **Dependencies**: 30+ direct dependencies in ruvector-graph including petgraph, roaring, nom, ordered-float, lru, moka.
- **Build dependencies**: pest_generator (for Cypher grammar).

**Risks**:
- **WASM binary size**: 30+ dependencies compiled to WASM could produce a very large binary. No wasm-opt or tree-shaking documented.
- **petgraph dependency**: Used for graph algorithms but not directly for the core GraphDB storage. May be pulling in unnecessary code.
- **No Windows ARM support**: Pre-built NAPI binaries listed for linux-x64, darwin-arm64, win32-x64. Windows ARM users have no pre-built option.
- **tokio version alignment**: Both ruvector-graph and ruvector-raft depend on tokio. Version mismatches could cause issues.

**Test Ideas**:
1. Measure WASM binary size (`wasm-pack build --release`) -- confirm it is under 5MB for acceptable web delivery. **P2**
2. Run NAPI tests on all three pre-built platforms (Linux x64, macOS ARM64, Windows x64) -- confirm bindings work. **P1**
3. Build ruvector-graph with `--features "wasm"` and without `--features "full"` -- confirm minimal WASM build compiles. **P1**
4. Load test the NAPI bindings from Node.js with 100 concurrent async operations -- confirm tokio runtime handles them without OOM or thread exhaustion. **P1**
5. Run graph operations in a Web Worker (WASM) -- confirm the WASM module works in a non-main-thread context. **P2**

---

### O -- Operations

**Current State**:
- **Statistics**: `GraphDB` exposes `node_count()`, `edge_count()`, `hyperedge_count()`. NAPI exposes `stats()` with totals and avg_degree. WASM exposes `stats()` with node/edge/hyperedge counts plus hypergraph stats.
- **Persistence**: Optional via `GraphDB::with_storage()`. Loads all data into memory on startup. Writes to redb on every mutation.
- **Query execution stats**: Executor pipeline collects per-step timing.
- **No admin operations**: No compaction, no index rebuild, no migration tools.
- **No logging/tracing integration in core GraphDB**: The distributed layer uses tracing, but the core graph operations do not emit structured logs.

**Risks**:
- **Full data in memory**: GraphDB loads ALL nodes, edges, and hyperedges into memory on startup. For large graphs (10M+ nodes), this could exhaust RAM.
- **No incremental loading**: No lazy loading or paging. Every graph operation reads from in-memory DashMaps, not storage.
- **MVCC version chains are never cleaned up**: TransactionManager has no garbage collection. Production use would require periodic restarts.
- **Query cache has no eviction policy documented**: The executor cache in `executor/cache.rs` may grow without bound.

**Test Ideas**:
1. Create a graph with 1M nodes and 5M edges, measure peak memory usage -- confirm it stays under 8GB (or document the memory model). **P1**
2. Run 100K transactions via TransactionManager, measure memory growth -- confirm it stabilizes or document the leak rate. **P1**
3. Persist a graph with 100K nodes to redb, measure `with_storage()` load time -- confirm it completes in under 30 seconds. **P2**
4. Execute the same Cypher query 1000 times -- confirm the query cache provides speedup on subsequent executions. **P2**
5. Monitor tracing output during distributed graph operations -- confirm structured logs include query IDs, shard IDs, and timing. **P2**

---

### T -- Time

**Current State**:
- **Transaction timestamps**: MVCC uses `SystemTime::now().duration_since(UNIX_EPOCH).as_micros() as u64` for version timestamps. No monotonic clock guarantee.
- **Transaction isolation**: 4 levels declared. MVCC visibility based on `created_at <= start_time && deleted_at > start_time`. The filter `v.created_by != txn_id` excludes the current transaction's own version-store writes (write-set takes precedence).
- **Concurrent access**: DashMap provides lock-free reads and per-shard writes. GraphDB create/delete operations update multiple DashMaps non-atomically.
- **Distributed timing**: Gossip membership, heartbeats, and RPC timeouts (in the distributed module). No clock synchronization mechanism documented.

**Risks**:
- **SystemTime can go backward**: `SystemTime::now()` is not monotonic. NTP corrections or clock adjustments could cause a later transaction to have an earlier timestamp, breaking MVCC visibility.
- **Non-atomic multi-DashMap updates**: `create_node()` calls `label_index.add_node()`, `property_index.add_node()`, `nodes.insert()`, and optionally `storage.insert_node()` in sequence. A concurrent reader between any two of these steps sees inconsistent state.
- **No deadlock detection**: The TransactionManager does not detect deadlocks between transactions. Two transactions could wait on each other's resources indefinitely (though current implementation does not have blocking waits, so this is theoretical with future enhancements).
- **WASM has no real concurrency**: WASM is single-threaded. All Mutex usage is effectively no-ops, but `parking_lot::Mutex` in WASM has undefined behavior if re-entered.

**Test Ideas**:
1. Mock `SystemTime::now()` to return decreasing values (simulating NTP correction), run two transactions, verify MVCC correctness -- confirm the system handles non-monotonic clocks. **P0**
2. Spawn 100 threads, each creating 100 nodes with the same label "Test", then query `get_nodes_by_label("Test")` -- confirm exactly 10,000 nodes are returned (no lost updates due to DashMap races). **P0**
3. Begin transaction T1, write node X, begin T2, write node X with different data, commit T1, commit T2 -- confirm T2's write is the surviving version (last-write-wins or conflict detection). **P1**
4. In the distributed graph, partition the network between two shards, send a cross-shard query -- confirm timeout and graceful error. **P1**
5. Measure the maximum transaction throughput (transactions/second) with RepeatableRead isolation -- establish a performance baseline. **P2**

---

## D3: Distributed Systems

### S -- Structure

**Current State**:
- **Raft consensus** (ruvector-raft): `RaftNode` orchestrates `PersistentState`, `VolatileState`, `LeaderState`, `ElectionState`. Messages flow through `mpsc::unbounded_channel`. `RaftLog` stores entries in `VecDeque<LogEntry>`. Election timer uses randomized timeouts (150-300ms default).
- **Replication** (ruvector-replication): `ReplicaSet` manages replicas with roles (Primary, Secondary, Arbiter, Hidden). `SyncManager` supports Sync, Async, SemiSync modes. `ConflictResolver` trait with `LastWriteWins`, `MergeFunction`, `MaxMerge`, `SetUnion` implementations. `FailoverManager` with health monitoring and automatic promotion. `VectorClock` for causality tracking.
- **Cluster** (ruvector-cluster): `ClusterManager` with `ConsistentHashRing` (150 virtual nodes per real node), `ShardRouter` (jump consistent hashing), `DagConsensus` (DAG-based alternative to Raft), `DiscoveryService` trait with `StaticDiscovery` and `GossipDiscovery`.
- **Delta system** (ruvector-delta-*): 5 sub-crates. `delta-core`: `VectorDelta` (compute, apply, compose, inverse), `DeltaStream`, `DeltaWindow`, compression/encoding. `delta-consensus`: `DeltaConsensus` with `CausalDelta`, `VectorClock`, `DeltaGossip`, CRDT implementations (GCounter, PNCounter, LWWRegister, ORSet). `delta-graph`, `delta-index`, `delta-wasm`.

**Risks**:
- **Two consensus mechanisms**: Raft (ruvector-raft) AND DAG consensus (ruvector-cluster::consensus). The Raft implementation has TODO comments for actual RPC sending. The DAG consensus lacks cross-node communication. Neither is production-ready for actual distributed deployment.
- **Raft RPC is unimplemented**: Lines 205-206, 475-476, 540 in raft/node.rs have `// TODO: Send request to member` and `// TODO: Send response back to sender`. The Raft node can process messages but cannot actually communicate with other nodes.
- **Snapshot installation is a stub**: Line 394 in raft/node.rs: `// TODO: Implement snapshot installation`. The `handle_install_snapshot()` always returns success.
- **`DeltaConsensus` has no network layer**: The gossip protocol (`DeltaGossip`) queues deltas in a `Vec<CausalDelta>` outbox but has no actual send mechanism. The `receive_gossip()` method must be called explicitly.
- **Deeply layered abstractions**: The distributed graph depends on ruvector-raft + ruvector-cluster + ruvector-replication. These three crates have overlapping but distinct concepts (Raft term vs. DAG vertex, ReplicaSet vs. ClusterNode, VectorClock in replication vs. VectorClock in delta-consensus).

**Test Ideas**:
1. Create a 3-node Raft cluster in-process (mocking the network layer), trigger an election, confirm a leader is elected with majority vote. **P0**
2. Create a `DeltaConsensus` with causal_delivery=true, send deltas out of order (D2 before D1, where D2 depends on D1) -- confirm D2 is queued until D1 is delivered. **P0**
3. Add and remove nodes from a `ConsistentHashRing`, measure key remapping -- confirm fewer than 1/N keys are remapped (consistent hashing guarantee). **P1**
4. Trigger failover in `FailoverManager` when no healthy secondaries exist -- confirm `QuorumNotMet` error, not panic. **P1**
5. Map the dependency graph of ruvector-raft, ruvector-cluster, ruvector-replication, ruvector-delta-consensus -- document where concepts overlap and which should be the canonical implementation. **P1**

---

### F -- Function

**Current State**:
- **Raft operations**: `submit_command()` appends to log and triggers replication. `handle_append_entries()` validates term, checks log consistency, appends entries, updates commit index. `handle_request_vote()` validates term, voted_for, log up-to-dateness.
- **Replication operations**: `ReplicaSet` add/remove/promote replicas. `SyncManager` with configurable sync modes. `ReplicationLog` with sequence-numbered, checksummed entries. `ReplicationStream` for change data capture.
- **Cluster operations**: `ClusterManager::add_node()` / `remove_node()` with automatic shard rebalancing. `assign_shard()` uses consistent hashing. `run_health_checks()` marks unhealthy nodes as Offline.
- **Delta operations**: `VectorDelta::compute(old, new)` produces delta, `apply()` reconstructs, `compose()` chains deltas, `inverse()` reverses. `DeltaStream` for event sourcing with checkpoints. `DeltaWindow` for time-bounded aggregation.
- **CRDT operations**: GCounter (increment, value, merge), PNCounter (increment, decrement, value), LWWRegister (set, get, merge with timestamp tiebreaker), ORSet (add, remove, contains, merge with tombstones).

**Risks**:
- **Raft log uses VecDeque**: In-memory only. No persistence. A node restart loses the entire Raft log. The Raft spec requires log persistence for safety.
- **Replication checksum uses DefaultHasher**: `LogEntry::calculate_checksum()` uses `std::collections::hash_map::DefaultHasher` which is NOT cryptographic and NOT deterministic across Rust versions. Log entries verified by checksum on one node may fail on another with a different Rust version.
- **FailoverManager promotes based on priority and lag**: But lag (`lag_ms`) is never updated by the `check_replica_health()` function -- it only checks `is_timed_out()` and `is_healthy()`. The lag metric is effectively always 0.
- **VectorClock `happens_before()` has a subtle bug**: At line 72, when `self` and `other` have all equal timestamps, the function returns `true` (because `equal` starts as `true` and `less` is `false`). But equal clocks should return `false` for `happens_before` -- they are equal, not one before the other. The `compare()` function handles this correctly because it checks `self == other` first, but `happens_before()` used independently returns `true` for equal clocks.
- **DagConsensus `finalize_vertices()` is O(V^2)**: For each vertex, it iterates all other vertices to count confirmations. With 100K vertices, this is 10B operations.

**Test Ideas**:
1. Create VectorClock A and B with identical values, call `a.happens_before(&b)` -- confirm it returns `true` (documenting the behavior) or `false` (if the intent is strict happens-before). **P0**
2. Create a Raft node, append 100 entries, simulate restart (drop and recreate) -- confirm the log is empty (documenting the non-persistence). **P0**
3. Create replicas with different `lag_ms` values, trigger failover -- confirm the candidate with lowest lag is selected. Then verify lag_ms is actually populated during health checks. **P1**
4. Create two `ReplicationLog` entries with identical data, verify checksums match across two separate instances -- confirm determinism. **P1**
5. Submit 10K transactions to DagConsensus, call `finalize_vertices()` -- measure wall-clock time and confirm it completes within 10 seconds (or identify the O(V^2) bottleneck). **P1**

---

### D -- Data

**Current State**:
- **Raft log entries**: `LogEntry { term: u64, index: u64, command: Vec<u8> }`. Generic bytes -- no schema.
- **Replication log entries**: `LogEntry { id: Uuid, sequence: u64, timestamp: DateTime<Utc>, data: Vec<u8>, checksum: u64, source_replica: String }`.
- **Cluster state**: `ClusterNode { node_id, address: SocketAddr, status, last_seen, metadata, capacity }`. `ShardInfo { shard_id: u32, primary_node, replica_nodes, vector_count, status }`.
- **Delta types**: `VectorDelta` with dense (Vec<f32>) and sparse (indices + values) representations. `CausalDelta` wraps `VectorDelta` with `VectorClock`, origin replica, timestamp, dependencies.
- **CRDT state**: `GCounter { counts: HashMap<ReplicaId, u64> }`, `PNCounter { positive: GCounter, negative: GCounter }`, `LWWRegister<T> { value, timestamp, replica }`, `ORSet<T> { elements: HashMap<T, HashSet<String>>, tombstones: HashSet<String> }`.

**Risks**:
- **ORSet tombstones grow unbounded**: Every `remove()` adds tombstone tags to a `HashSet<String>` that is never compacted. Over time, the tombstone set dominates memory.
- **VectorDelta dimension validation**: `VectorDelta::apply()` presumably checks that the delta dimensions match the base vector, but this depends on the implementation. Mismatched dimensions could corrupt data.
- **CausalDelta timestamp is wall-clock**: `chrono::Utc::now().timestamp_millis() as u64`. Wall-clock timestamps can go backward, have duplicate values across replicas, and are not totally ordered.
- **Replication Log has no compaction**: Entries accumulate in `DashMap<u64, LogEntry>` with no eviction. Long-running replicas will exhaust memory.
- **Shard state is ephemeral**: ShardInfo is stored in DashMap, not persisted. A cluster restart loses all shard assignments.

**Test Ideas**:
1. Add and remove 100K elements from an ORSet, measure memory usage -- confirm tombstone growth and document the accumulation rate. **P1**
2. Compute a VectorDelta between a 384-dim and 768-dim vector -- confirm error or panic, then apply a 384-dim delta to a 768-dim base -- confirm dimension validation. **P0**
3. Create `CausalDelta` instances with identical timestamps from different replicas -- confirm the conflict resolution strategy produces deterministic results regardless of delivery order. **P0**
4. Append 1M entries to a `ReplicationLog`, measure memory consumption -- confirm it stays under 1GB for reasonably-sized entries. **P1**
5. Persist and restore cluster shard assignments -- confirm that after restart, shard routing produces the same results (or document that shard state is ephemeral). **P1**

---

### I -- Interfaces

**Current State**:
- **Raft interface**: `RaftNode::submit_command(data: Vec<u8>)` returns `CommandResult { index, term }`. Message handling via internal `mpsc` channel. External RPC is TODO.
- **Replication interface**: `ReplicaSet::add_replica()`, `SyncManager::set_sync_mode()`, `ConflictResolver::resolve()` trait. `ReplicationStream` for CDC.
- **Cluster interface**: `ClusterManager::new()` takes a `Box<dyn DiscoveryService>`. `add_node()`/`remove_node()` are async. `ShardRouter::get_shard()` returns shard ID.
- **Delta interface**: `DeltaConsensus::create_delta()` / `receive()`. `DeltaGossip::broadcast()` / `receive_gossip()`. All in-process, no network transport.
- **Error types**: Each crate has its own error type: `RaftError` (10 variants), `ReplicationError` (10 variants), `ClusterError` (7 variants), `ConsensusError` (from delta-consensus), `DeltaError` (from delta-core). No unified error hierarchy.

**Risks**:
- **No actual network transport**: All distributed crates are libraries with no built-in networking. Raft, Cluster, Replication, and Delta consensus all require the integrator to implement actual RPC/message passing. This is a huge integration gap.
- **Error types are incompatible**: `RaftError`, `ReplicationError`, `ClusterError`, `ConsensusError`, and `DeltaError` have no `From` conversions between them. An application using all three must handle 5 different error types.
- **Discovery service trait is simple**: Only has `discover_nodes()` returning `Vec<ClusterNode>`. No support for dynamic membership changes, health events, or deregistration callbacks.
- **Raft and DagConsensus have different interfaces**: They solve the same problem (consensus) with incompatible APIs. No adapter or abstraction to swap between them.

**Test Ideas**:
1. Implement a mock network transport for Raft (in-memory channel between 3 RaftNode instances), run a complete election and log replication cycle -- confirm end-to-end correctness. **P0**
2. Create a unified error wrapper that converts between all 5 error types -- confirm all error information is preserved through the conversion chain. **P2**
3. Implement `DiscoveryService` with a `GossipDiscovery` that uses UDP multicast, start 3 nodes -- confirm they discover each other within 10 seconds. **P1**
4. Wire `DeltaGossip` with a TCP transport between 2 nodes, create deltas on node 1, verify they appear on node 2 -- confirm end-to-end delta propagation. **P1**
5. Create a `ClusterManager`, call `start()`, then `remove_node()` for a non-existent node -- confirm error handling (currently returns Ok after no-op hash ring removal). **P2**

---

### P -- Platform

**Current State**:
- **Raft**: Depends on tokio (async runtime). Not WASM-compatible.
- **Cluster**: Depends on tokio. Not WASM-compatible.
- **Replication**: Depends on tokio, parking_lot, dashmap. Not WASM-compatible.
- **Delta-core**: `#![cfg_attr(not(feature = "std"), no_std)]` -- supports no_std environments. Uses `extern crate alloc`.
- **Delta-consensus**: Depends on parking_lot, chrono, uuid. Requires std.
- **Delta-wasm**: Provides WASM bindings via wasm-bindgen. Includes SIMD support (`delta-wasm/src/simd.rs`).
- **Platform constraints**: All distributed crates require std and tokio. Only delta-core is no_std compatible.

**Risks**:
- **tokio version coupling**: All async crates depend on tokio. The workspace must maintain version alignment. A tokio major version bump cascades through all distributed crates.
- **No embedded/no_std distributed system**: The distributed layer cannot run on embedded devices or in WASM (except delta-core). This limits edge computing use cases.
- **delta-wasm SIMD**: WASM SIMD support varies by browser. Fallback paths must be tested.
- **Chrono dependency in multiple crates**: chrono is used in cluster (ClusterNode.last_seen), replication (LogEntry.timestamp, HealthCheck.timestamp), and delta-consensus (CausalDelta.timestamp). Chrono has had security advisories (localtime_r unsoundness).

**Test Ideas**:
1. Build delta-core with `--no-default-features` (no_std mode) -- confirm it compiles for `thumbv7em-none-eabihf` (ARM embedded target). **P1**
2. Build delta-wasm, run in Chrome/Firefox/Safari -- confirm WASM SIMD delta operations produce correct results. **P1**
3. Run the full distributed test suite with tokio 1.x and verify it compiles -- confirm no deprecated API usage. **P2**
4. Audit chrono usage for the `localtime_r` unsoundness (CVE-2020-26235) -- confirm only UTC functions are used (they are: `Utc::now()`). **P1**
5. Build ruvector-raft for wasm32-unknown-unknown -- confirm it fails to compile (documenting the platform limitation). **P2**

---

### O -- Operations

**Current State**:
- **Raft monitoring**: `RaftNode` exposes `current_state()`, `current_term()`, `current_leader()`. No metrics export.
- **Cluster monitoring**: `ClusterManager::get_stats()` returns `ClusterStats { total_nodes, healthy_nodes, total_shards, active_shards, total_vectors }`. `run_health_checks()` marks unhealthy nodes.
- **Replication monitoring**: `FailoverManager::health_history()` returns last 1000 health checks. `failure_count()` per replica.
- **Consensus monitoring**: `DagConsensus::get_stats()` returns vertex/finalized/pending/tip counts. `DeltaConsensus::pending_count()`.
- **No operational tooling**: No CLI for cluster management, no admin API, no migration tools, no backup/restore.
- **Configuration**: All via struct fields. No file-based configuration. No runtime reconfiguration.

**Risks**:
- **No cluster recovery procedure**: If the Raft leader crashes and the log is lost (in-memory only), the cluster cannot recover. There is no documented recovery procedure.
- **Health check interval is fixed at construction**: `ClusterConfig::heartbeat_interval` and `FailoverPolicy::health_check_interval` cannot be changed at runtime.
- **No alerting**: Health degradation is logged via tracing but not surfaced to an alerting system.
- **Shard rebalancing blocks on async**: `rebalance_shards()` iterates all shards sequentially within an async context. For 64 shards (default), this is fast, but for 1000+ shards, it could hold locks too long.

**Test Ideas**:
1. Start a 3-node cluster, kill the leader, confirm a new leader is elected AND the cluster resumes accepting writes -- end-to-end leader failover. **P0**
2. Start a `FailoverManager`, simulate 3 consecutive health check failures for the primary -- confirm automatic failover to the best secondary. **P0**
3. Start a cluster with 256 shards, add/remove nodes rapidly (10 per second) -- measure rebalancing throughput and confirm no shard is left unassigned. **P1**
4. Export cluster stats to Prometheus format -- confirm operational dashboards can visualize node health, shard distribution, and replication lag. **P2**
5. Simulate a split-brain scenario (two network partitions each with a leader) -- confirm `prevent_split_brain` in FailoverPolicy prevents dual-primary. **P1**

---

### T -- Time

**Current State**:
- **Raft timing**: Election timeout 150-300ms (randomized). Heartbeat interval 50ms. Election timer polls every 50ms. All via tokio::time.
- **Cluster timing**: Heartbeat interval 5s, node timeout 30s (defaults in ClusterConfig).
- **Replication timing**: Health check interval 5s, timeout 2s, failure threshold 3 (defaults in FailoverPolicy).
- **Delta timing**: CausalDelta timestamps via `chrono::Utc::now().timestamp_millis()`. HybridLogicalClock available in delta-consensus.
- **Vector clocks**: Used in replication (conflict.rs) and delta-consensus (causal.rs). Two independent implementations.

**Risks**:
- **Election timer is polled, not event-driven**: The election timer task runs every 50ms (`interval(Duration::from_millis(50))`), checking `should_start_election()`. This means election timeout resolution is 50ms. In a busy system with tokio executor delays, actual election start could be 50-100ms late.
- **No clock synchronization**: The cluster assumes wall-clock timestamps are roughly synchronized. No NTP integration, no HLC in the main cluster/raft code (HLC exists only in delta-consensus).
- **Race condition in FailoverManager**: `trigger_failover()` uses `failover_in_progress` flag (RwLock<bool>) as a guard. Between checking `*in_progress` and setting it to `true`, another thread could also enter. However, the current code does this atomically within a single write lock scope, so this is safe.
- **`pending` counter in LockFreeBatchProcessor is non-decreasing**: `pending()` is incremented on `submit()` but never decremented. `is_done()` checks `pending() == completed()`, which only works if no new items are submitted. The comment "Check if all work is done" is misleading.
- **Two VectorClock implementations**: `ruvector_replication::conflict::VectorClock` and `ruvector_delta_consensus::causal::VectorClock` are independent. They may have different `happens_before` semantics (the replication one returns `true` for equal clocks, as noted above).

**Test Ideas**:
1. Create a 5-node Raft cluster, introduce 200ms network latency between all nodes, run for 60 seconds -- confirm stable leadership (no constant re-elections due to tight timeouts). **P0**
2. Run the Raft election timer with a heavily loaded tokio runtime (1000 concurrent tasks) -- measure actual election timeout vs. configured timeout, confirm jitter is under 100ms. **P1**
3. Create two `ruvector_replication::VectorClock` instances with identical values, compare using `happens_before()` and `compare()` -- confirm `happens_before` returns `true` but `compare` returns `Equal` (documenting the semantic difference). **P0**
4. Submit 10K deltas to `DeltaConsensus` with causal_delivery enabled and random ordering -- measure delivery latency and confirm all deltas are eventually delivered in causal order. **P1**
5. Run a 24-hour soak test of the full distributed stack (cluster + raft + replication) with periodic node failures -- measure time-to-recovery and data consistency. **P1**

---

## Cross-Domain Risks

| Risk | Domains | Priority | Description |
|------|---------|----------|-------------|
| **No end-to-end integration test** | D1 + D2 + D3 | P0 | No test exercises the full path: client -> graph query -> vector search -> distributed consensus -> replication. Each domain is tested in isolation. |
| **Two graph systems** | D2 | P0 | `GraphDB` (native Rust) and ruvector-graph-wasm `GraphDB` are completely separate implementations. Feature parity is not maintained. |
| **No network transport** | D3 | P0 | All distributed crates lack actual networking. Raft, cluster, and replication are library-only with no runnable distributed system. |
| **VectorClock semantic mismatch** | D3 | P0 | Two independent VectorClock implementations with different `happens_before` semantics for equal clocks. |
| **In-memory Raft log** | D3 | P0 | Raft log is not persistent. A single node restart violates Raft's safety guarantees. |
| **HNSW delete is a no-op** | D1 | P0 | Deleted vectors remain in the HNSW graph and can appear as ghost results. |
| **MVCC has no GC** | D2 | P1 | Version chains grow without bound. Long-running graph workloads leak memory. |
| **ORSet tombstones unbounded** | D3 | P1 | Tombstone accumulation in ORSet CRDT has no compaction mechanism. |
| **Global DB_POOL leak** | D1 | P1 | The static database connection pool never evicts entries. |
| **Feature flag explosion** | D1 + D2 | P1 | ruvector-core (8 flags) x ruvector-graph (14 flags) = 22 flags, not all combinations tested. |

---

## Test Idea Summary

| Priority | Count | Domain Distribution |
|----------|-------|---------------------|
| **P0** | 19 | D1: 6, D2: 7, D3: 6 |
| **P1** | 37 | D1: 12, D2: 12, D3: 13 |
| **P2** | 24 | D1: 8, D2: 8, D3: 8 |
| **Total** | **80** | |

### Automation Fitness

| Category | Count | Percentage |
|----------|-------|------------|
| Unit tests (Rust `#[test]`) | 28 | 35% |
| Integration tests (multi-crate, cargo test) | 22 | 28% |
| Property-based tests (proptest) | 8 | 10% |
| Performance/benchmark tests | 10 | 12% |
| Human exploration (architecture review, manual inspection) | 12 | 15% |

---

## Clarifying Questions

These questions surfaced from gaps discovered during the code analysis. They are suggestions based on general risk patterns observed in the codebase.

### D1: Core Vector DB

1. **What is the intended behavior of HNSW delete?** The code acknowledges it as a "known limitation." Is the plan to rebuild the index periodically, use a different ANN library, or implement soft-delete filtering at query time?
2. **Is NaN/Inf validation intentionally omitted?** Some vector databases validate floating-point inputs. Is there a performance reason to skip this?
3. **What is the target maximum database size (vectors)?** The current architecture loads the full HNSW index into memory. At what scale does this become a problem?
4. **Should metadata filters be pushed down to the index?** Currently, post-hoc filtering reduces result quality. Is pre-filtering or over-fetching planned?

### D2: Graph Database

5. **Are `GraphDB` and `TransactionManager` intended to be separate systems?** They appear to be parallel implementations that don't share state. Should transactions be integrated into GraphDB?
6. **Should node deletion cascade to edges?** Current behavior leaves orphaned edges. Is this intentional (for soft-delete patterns) or a missing feature?
7. **Why does ruvector-graph-wasm have its own GraphDB implementation?** This creates a maintenance burden. Is consolidation planned?
8. **What Cypher compliance level is targeted?** The parser handles many clauses but may have gaps in complex expressions, aggregation, and path patterns.

### D3: Distributed Systems

9. **Which consensus mechanism is canonical -- Raft or DAG?** Both exist but neither is complete. Which should be invested in?
10. **When will Raft RPC transport be implemented?** The Raft node logic is sound but cannot communicate. Is tonic/gRPC the intended transport?
11. **Is the Raft log intentionally in-memory?** The Raft spec requires persistent logs for safety. Is this a known gap with a planned fix?
12. **Should VectorClock implementations be unified?** Two independent implementations with different semantics is a correctness hazard.

---

*Analysis generated by SFDIPOT Product Factors Assessor. All findings based on actual code review of the RuVector monorepo at commit e9bbc7de.*
