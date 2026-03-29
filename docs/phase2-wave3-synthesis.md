# Phase 2 Wave 3+4: P2/P3 Domains — Synthesis Report

**Date**: 2026-03-29
**Status**: COMPLETE (all 4 agents finished)
**Scope**: D6 WASM, D7 Node.js, D8 CLI & Router, D9 UI, D10 Specialized

---

## Executive Summary

The P2/P3 domains reveal a pattern of **incomplete implementations masked by clean APIs**. The WASM layer has 33 crates but many are stubs returning empty results silently — notably the entire graph async layer is a facade. The Node.js bindings have strong architecture (zero unsafe, good async) but zero memory leak tests and cascading lock-poisoning risks. The CLI is the cleanest domain with proper error handling and good test coverage. The UI has solid auth (OIDC+PKCE) but an XSS vector in markdown rendering. D10 has the monorepo's worst file (6,807 LOC) and orphaned crates.

---

## Domain Scorecards

### D6: WASM Bindings — 40/100 (Poor)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Code Quality | 45 | 33 crates, well-structured, but many stubs |
| Code Smells | 30 | Silent stubs, commit() no-ops, async facades |
| Security | 50 | Zero unsafe outside SIMD, auto-generated free(), 1 debug_assert bug |
| Performance | 40 | BufferPool.release() no-op → unbounded memory |
| QX | 35 | 22/33 crates have no TypeScript definitions |
| Test Coverage | 35 | 536 tests total but 10 crates with zero tests |
| Architecture | 55 | Clean wasm-bindgen patterns, proper error propagation |

### D7: Node.js Bindings — 42/100 (Poor)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Code Quality | 55 | 9,231 LOC, well-structured NAPI crates |
| Code Smells | 35 | 37 lock unwraps, wrong-platform binaries |
| Security | 45 | Zero unsafe in 9/10 crates, but no memory leak tests |
| Performance | 50 | Good async patterns with spawn_blocking |
| QX | 40 | 14 `any` types defeat TypeScript safety |
| Test Coverage | 20 | 8/10 packages have zero JS tests |
| Architecture | 60 | 5-platform binaries, consistent error propagation |

### D8: CLI & Router — 62/100 (Acceptable)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Code Quality | 55 | hooks.rs needs splitting, otherwise clean |
| Code Smells | 60 | 2 command stubs, silent Cosine default |
| Security | 55 | Good MCP path protection, but CLI args unvalidated, ruvllm CORS open |
| Performance | 70 | HNSW search engine is efficient |
| QX | 70 | Good error messages, proper exit codes, debug mode |
| Test Coverage | 60 | 29 tests in ruvector-cli, 0 in ruvllm-cli |
| Architecture | 65 | Clean separation, proper deadlock fix with regression test |

### D9: UI Layer — 50/100 (Acceptable)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Code Quality | 55 | SvelteKit, well-organized components |
| Code Smells | 50 | Only 1 component test, limited CSP |
| Security | 45 | OIDC+PKCE auth, but XSS in markdown, MCP secret fallback |
| Performance | 55 | Standard SvelteKit patterns |
| QX | 60 | 36 aria attributes, decent a11y |
| Test Coverage | 25 | 1 Svelte component test out of dozens |
| Architecture | 60 | Clean component structure, proper cookie config |

### D10: Specialized/Research — 35/100 (Poor)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Code Quality | 25 | 6,807 LOC routes.rs, many oversized files |
| Code Smells | 30 | Orphaned crates, workspace exclude contradictions |
| Security | 40 | RuVix unsafe justified, but routes.rs unaudited |
| Performance | 40 | No benchmarks for brain server |
| QX | 35 | 84 handlers in one file, poor discoverability |
| Test Coverage | 30 | Brain server has tests but orphaned crates have none |
| Architecture | 45 | RuVix well-designed, but brain server coupling |

---

## Cross-Wave Patterns Confirmed

### 1. Silent Stub Pattern (NEW — Wave 3)
Multiple crates across D6 and D7 have functions that **appear to work but silently return empty/default results**:
- Graph WASM: Cypher MATCH/CREATE return empty
- Graph WASM: AsyncTransaction.commit() succeeds without executing
- Graph WASM: All 5 async ops return null
- IVFFlat: INSERT doesn't actually insert (D4)
- HNSW: remove() doesn't actually remove (D1)

This is a pervasive pattern — functions exist to satisfy an API contract but don't implement behavior.

### 2. `debug_assert!` Systemic Bug (CONFIRMED across 4 domains)
Now found in D1, D4, D5, D6 — approximately 95+ sites across 50+ functions.

### 3. CORS `allow_origin(Any)` Pattern (3 occurrences)
- ruvector-server (D4)
- ruvllm serve (D8)
- ruvllm-wasm (D6, if applicable)

### 4. Lock Poisoning Cascade Risk
- D2: 27 RwLock::write().unwrap() calls
- D7: 37 expect("RwLock poisoned")/unwrap() calls
- Total: 64+ sites where one panic cascades into all subsequent operations

---

## Wave 3+4 Deliverables

| Report | Location |
|--------|----------|
| D6 WASM Bindings | `docs/phase2-d6-wasm-bindings.md` |
| D7 Node.js Bindings | `docs/phase2-d7-nodejs-bindings.md` |
| D8 CLI & Router | `docs/phase2-d8-cli-router.md` |
| D9+D10 UI + Specialized | `docs/phase2-d9d10-ui-specialized.md` |
| **This Synthesis** | `docs/phase2-wave3-synthesis.md` |

---

## Complete Domain Scorecard (All 10 Domains)

| Domain | Score | Rating | Priority | Weight |
|--------|-------|--------|----------|--------|
| D1 Core Vector DB | 52 | Acceptable | P0 | 3x |
| D2 Graph Database | 38 | Poor | P0 | 3x |
| D3 Distributed Systems | 18 | Critical | P0 | 3x |
| D4 Security & Persistence | 22 | Critical | P1 | 2x |
| D5 Neural/ML | 48 | Poor-Acceptable | P1 | 2x |
| D6 WASM Bindings | 40 | Poor | P2 | 1x |
| D7 Node.js Bindings | 42 | Poor | P2 | 1x |
| D8 CLI & Router | 62 | Acceptable | P2 | 1x |
| D9 UI Layer | 50 | Acceptable | P2 | 1x |
| D10 Specialized | 35 | Poor | P3 | 1x |

**Weighted Overall Score**: 37.9/100 (Poor)

Formula: (52×3 + 38×3 + 18×3 + 22×2 + 48×2 + 40 + 42 + 62 + 50 + 35) / (3+3+3+2+2+1+1+1+1+1) = 37.9
