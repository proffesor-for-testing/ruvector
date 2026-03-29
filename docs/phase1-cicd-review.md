# Phase 1 -- Step 1.10: CI/CD Workflow Review

**Date**: 2026-03-29
**Scope**: All files in `.github/workflows/`
**Branch analyzed**: `qe-working-branch` (based on `main`)

---

## 1. Workflow Inventory

The repository contains **29 workflow YAML files** (plus 2 markdown docs). Each is summarized below with its name, triggers, and purpose.

| # | File | Workflow Name | Triggers | Purpose |
|---|------|--------------|----------|---------|
| 1 | `agentic-synth-ci.yml` | Agentic-Synth CI/CD | push(main/develop/claude/**), PR(main/develop), dispatch | Full CI for `packages/agentic-synth` npm package: lint, type-check, build, test (3 OS x 3 Node), coverage, security audit, package validation, docs check |
| 2 | `benchmarks.yml` | Benchmarks | PR(postgres paths), push(main/develop), dispatch | Rust benchmarks for `ruvector-postgres`: distance, index, quantization. SQL benchmarks (dispatch-only). PR comparison against baseline |
| 3 | `build-attention.yml` | Build Attention Native Modules | push(main, v* tags, attention paths), PR(main, attention paths), dispatch | Cross-platform NAPI-RS build for `ruvector-attention-node` + WASM build for `ruvector-attention-wasm`. Commit binaries, publish on tag |
| 4 | `build-gnn.yml` | Build GNN Native Modules | push(main, v* tags, gnn paths), PR(main, gnn paths), dispatch | Cross-platform NAPI-RS build for `ruvector-gnn-node` (7 targets incl. musl). Commit binaries, publish on tag/dispatch |
| 5 | `build-graph-node.yml` | Build Graph Node Native Modules | push(main, v* tags, graph paths), PR(main, graph paths), dispatch | Cross-platform NAPI-RS build for `ruvector-graph-node`. Publish on tag/dispatch |
| 6 | `build-graph-transformer.yml` | Build Graph Transformer Native Modules | push(main, v* tags, transformer paths), PR(main, transformer paths), dispatch | Cross-platform NAPI-RS + WASM build for `ruvector-graph-transformer-node`. Commit binaries, publish on tag/dispatch |
| 7 | `build-native.yml` | Build Native Modules | push(main, v* tags), PR(main), dispatch, workflow_call | Core NAPI-RS build for `npm/packages/core` (ruvector-node). Commit binaries |
| 8 | `build-router.yml` | Build Router Native Modules | push(main, v* tags, router paths), PR(main, router paths), dispatch | Cross-platform NAPI-RS build for `ruvector-router-ffi`. Publish on tag/dispatch |
| 9 | `build-rvf-node.yml` | Build RVF Node Native Modules | push(main, rvf paths), PR(main, rvf paths), dispatch | Cross-platform NAPI-RS build for `rvf-node`. Commit binaries |
| 10 | `build-tiny-dancer.yml` | Build Tiny Dancer Native Modules | push(main, v* tags, tiny-dancer paths), PR(main, tiny-dancer paths), dispatch | Cross-platform NAPI-RS build for `ruvector-tiny-dancer-node`. Publish on tag/dispatch |
| 11 | `build-verified.yml` | ruvector-verified CI | push(verified paths), PR(verified paths) | Feature-matrix check, test, bench dry-run, clippy for `ruvector-verified` |
| 12 | `copilot-setup-steps.yml` | Copilot Setup Steps | workflow_call | Setup for GitHub Copilot coding agent (Node.js, ruvector install) |
| 13 | `docker-publish.yml` | Docker Hub Publish | release(published), dispatch | Multi-arch Docker images for `ruvector-postgres` (PG 14-17). Publish to Docker Hub + GHCR |
| 14 | `edge-net-models.yml` | Edge-Net Model Optimization | release(published), dispatch | ONNX model quantization (INT4/INT8/FP16), upload to GCS/IPFS, registry update, benchmarks |
| 15 | `hooks-ci.yml` | Hooks CI | push(main/claude/*, hooks paths), PR(main, cli paths) | Rust CLI hooks tests, npm CLI hooks tests, PostgreSQL schema validation, feature parity check |
| 16 | `postgres-extension-ci.yml` | PostgreSQL Extension CI | push(main/develop/claude/fix, postgres paths), PR(main/develop, postgres paths), dispatch | Build/test PG extension (PG17, Ubuntu+macOS), all-features test, benchmark, security audit (`cargo audit`), packaging, Docker integration |
| 17 | `publish-all.yml` | Build & Publish All Packages | push(v* tags), dispatch | Full release: validate, build native (5 platforms), build WASM (math+attention), publish to crates.io + npm, create GitHub Release |
| 18 | `release-rvf-cli.yml` | Release RVF CLI | push(rvf-v* tags), dispatch | Build RVF CLI binaries (5 platforms), create GitHub Release with checksums |
| 19 | `release.yml` | Release Pipeline | push(v* tags), dispatch | Full release pipeline: validate (fmt, clippy, workspace test), build crates, build WASM (4 packages), build native (reuses build-native), publish crates.io, prepare npm, create GitHub Release |
| 20 | `ruvector-postgres-ci.yml` | RuVector-Postgres CI/CD | push(main/develop/feat/claude/fix, postgres paths), PR(main/develop, postgres paths), dispatch | Comprehensive postgres CI: lint/fmt, test matrix (PG17 Ubuntu+macOS), all-features test, Docker integration, benchmark, security audit, package |
| 21 | `ruvllm-benchmarks.yml` | RuvLLM Benchmarks | PR(ruvllm paths), push(main/develop, ruvllm paths), dispatch | macOS ARM64 ANE benchmarks, Linux NEON benchmarks, cross-platform comparison |
| 22 | `ruvllm-build.yml` | RuvLLM Build & Publish | push(ruvllm-v* tags), dispatch | Build ruvllm native binaries (5 platforms incl. Docker), publish npm, test installation |
| 23 | `ruvllm-native.yml` | RuvLLM Native Build | push(ruvllm-v* tags), dispatch | Alternative ruvllm native build using napi-rs CLI directly. Publish platform packages |
| 24 | `ruvltra-tests.yml` | RuvLTRA-Small Tests | push(main/develop, ruvllm paths), PR(main/develop, ruvllm paths), dispatch | Comprehensive ruvllm testing: unit (3 OS), E2E (2 OS), Apple Silicon, quantization, thread safety, code quality (fmt+clippy+docs), coverage (llvm-cov, 60% threshold) |
| 25 | `sona-napi.yml` | SONA NAPI Build & Publish | push(sona-v* tags, sona paths), PR(sona paths), dispatch | SONA native build (7 targets + universal macOS), publish platform packages, test installation |
| 26 | `sync-rvf-examples.yml` | Sync RVF Examples to GCS | push(main, rvf example paths), dispatch | Sync RVF example files to Google Cloud Storage with manifest and checksums |
| 27 | `thermorust-ci.yml` | thermorust CI | push(thermorust paths), PR(thermorust paths) | Test (3 OS), fmt, clippy, bench compile check for `thermorust` crate |
| 28 | `validate-lockfile.yml` | Validate Package Lock File | PR(npm lock paths), push(main/develop, npm lock paths) | Validates npm `package-lock.json` exists, version, and name match |
| 29 | `wasm-dedup-check.yml` | WASM Dedup Check | push(main), PR(main) | Checks for duplicate WASM artifacts in node_modules |

---

## 2. Coverage Mapping

### 2.1 Crates with Dedicated CI Workflows

| Crate(s) | Workflow(s) | What's Tested |
|-----------|------------|---------------|
| `ruvector-postgres` | `postgres-extension-ci.yml`, `ruvector-postgres-ci.yml`, `benchmarks.yml`, `docker-publish.yml` | Build, pgrx tests, all-features, clippy, fmt, cargo audit, Docker integration, benchmarks, packaging |
| `ruvector-attention`, `ruvector-attention-node`, `ruvector-attention-wasm` | `build-attention.yml`, `publish-all.yml` | NAPI-RS cross-platform build, WASM build, module loading test |
| `ruvector-gnn`, `ruvector-gnn-node` | `build-gnn.yml` | NAPI-RS cross-platform build (7 targets) |
| `ruvector-graph`, `ruvector-graph-node` | `build-graph-node.yml` | NAPI-RS cross-platform build, npm test |
| `ruvector-graph-transformer`, `ruvector-graph-transformer-node`, `ruvector-graph-transformer-wasm` | `build-graph-transformer.yml` | NAPI-RS cross-platform build, WASM build |
| `ruvector-node` (core) | `build-native.yml` | NAPI-RS cross-platform build, module loading test |
| `ruvector-router-core`, `ruvector-router-ffi` | `build-router.yml` | NAPI-RS cross-platform build, npm test |
| `ruvector-tiny-dancer-core`, `ruvector-tiny-dancer-node` | `build-tiny-dancer.yml` | NAPI-RS cross-platform build, npm test |
| `ruvector-verified`, `ruvector-verified-wasm` | `build-verified.yml` | Feature-matrix cargo check, tests, bench dry-run, clippy |
| `ruvector-cli` | `hooks-ci.yml` | Hooks unit tests, command tests, feature parity with npm CLI |
| `ruvector-math`, `ruvector-math-wasm` | `publish-all.yml` | Tests (as part of validate), WASM build |
| `thermorust` | `thermorust-ci.yml` | Tests (3 OS), fmt, clippy, bench compile check |
| `ruvllm`, `ruvllm-cli` | `ruvltra-tests.yml`, `ruvllm-benchmarks.yml`, `ruvllm-build.yml`, `ruvllm-native.yml` | Unit tests (3 OS), E2E, Apple Silicon, quantization, thread safety, fmt, clippy, doc check, coverage, benchmarks, native builds |
| `sona` | `sona-napi.yml` | NAPI-RS build (7 targets + universal macOS), post-publish installation test |
| `rvf-node` (under `crates/rvf/`) | `build-rvf-node.yml` | NAPI-RS cross-platform build, module loading test |
| `rvf-cli` (under `crates/rvf/`) | `release-rvf-cli.yml` | Cross-platform binary build, GitHub Release |

### 2.2 Crates Tested Only via `--workspace` Builds

The `release.yml` workflow runs `cargo test --workspace --all-features` and `cargo clippy --workspace`. This covers many workspace members that lack dedicated workflows, including:

- `ruvector-core`
- `ruvector-bench`
- `ruvector-metrics`
- `ruvector-filter`
- `ruvector-snapshot`
- `ruvector-collections`
- `ruvector-cluster`
- `ruvector-raft`
- `ruvector-replication`
- `ruvector-server`
- `ruvector-router-cli`
- `ruvector-router-wasm`
- `ruvector-wasm`
- `ruvector-gnn-wasm`
- `ruvector-graph-wasm`
- `ruvector-tiny-dancer-wasm`
- `ruvector-cnn`, `ruvector-cnn-wasm`
- `ruvector-mincut`, `ruvector-mincut-wasm`, `ruvector-mincut-node`
- `ruvector-mincut-gated-transformer`, `ruvector-mincut-gated-transformer-wasm`
- `ruvector-nervous-system`, `ruvector-nervous-system-wasm`
- `ruvector-dag`, `ruvector-dag-wasm`
- `ruvector-economy-wasm`
- `ruvector-learning-wasm`
- `ruvector-exotic-wasm`
- `ruvector-attention-unified-wasm`
- `ruvector-fpga-transformer`, `ruvector-fpga-transformer-wasm`
- `ruvector-sparse-inference`, `ruvector-sparse-inference-wasm` (note: not listed in workspace members)
- `ruvector-math`
- `cognitum-gate-kernel`, `cognitum-gate-tilezero`
- `mcp-gate`, `mcp-brain`
- `ruQu`, `ruqu-core`, `ruqu-algorithms`, `ruqu-wasm`, `ruqu-exotic`
- `ruvllm-wasm`
- `prime-radiant`
- `ruvector-delta-core`, `ruvector-delta-wasm`, `ruvector-delta-index`, `ruvector-delta-graph`, `ruvector-delta-consensus`
- `ruvector-crv`
- `ruvector-temporal-tensor`
- `ruvector-domain-expansion`, `ruvector-domain-expansion-wasm`
- `ruvector-solver`, `ruvector-solver-wasm`, `ruvector-solver-node`
- `ruvector-coherence`
- `ruvector-profiler`
- `ruvector-attn-mincut`
- `ruvector-cognitive-container`
- `ruvector-dither`
- `ruvector-robotics`
- `neural-trader-core`, `neural-trader-coherence`, `neural-trader-replay`, `neural-trader-wasm`
- `rvlite`
- `ruvix/*` (types, region, queue, cap, proof, sched, boot, vecgraph, nucleus, hal, aarch64, drivers, tests, benches, cognitive_demo)
- `rvAgent/*` (rvagent-core, rvagent-backends, rvagent-middleware, rvagent-tools, rvagent-subagents, rvagent-cli, rvagent-acp, rvagent-mcp, rvagent-wasm)
- `ruvector-sparsifier`, `ruvector-sparsifier-wasm`

**Important caveat**: The workspace `--all-features` build only runs in the `release.yml` pipeline (triggered by `v*` tags or manual dispatch), NOT on every push or PR. Therefore, **on regular development pushes and PRs, most of these crates have NO CI coverage at all.**

### 2.3 Crates with NO CI Coverage

The following crates exist on disk but are **excluded from the workspace** in `Cargo.toml` and have **no dedicated workflows**:

| Crate | Status |
|-------|--------|
| `micro-hnsw-wasm` | Excluded from workspace, no CI |
| `ruvector-hyperbolic-hnsw` | Excluded from workspace, no CI |
| `ruvector-hyperbolic-hnsw-wasm` | Excluded from workspace, no CI |
| `mcp-brain-server` | Listed in workspace but also in exclude list (contradictory), no CI |
| `rvf/*` (entire RVF sub-workspace) | Excluded from workspace, no CI (only `rvf-node` and `rvf-cli` have CI via dedicated build/release workflows) |
| `agentic-robotics-*` (6 crates) | Not in workspace members, no CI |
| `ruvector-mincut-brain-node` | Not in workspace members, no CI |

Additionally, these crates are in the workspace but have no path-filtered CI (only get tested on release tags):

| Crate | Gap |
|-------|-----|
| `ruvector-core` | No per-push/PR CI despite being the foundational crate |
| `ruvector-cnn`, `ruvector-cnn-wasm` | No dedicated CI |
| `ruvector-mincut*` (5 crates) | No dedicated CI |
| `ruvector-nervous-system*` | No dedicated CI |
| `ruvector-solver*` (3 crates) | No dedicated CI |
| `ruvector-delta-*` (5 crates) | No dedicated CI |
| `ruQu*` (5 crates) | No dedicated CI |
| `ruvix/*` (15 crates) | No dedicated CI |
| `rvAgent/*` (9 crates) | No dedicated CI |
| `neural-trader-*` (4 crates) | No dedicated CI |
| `ruvector-sparsifier*` | No dedicated CI |
| `cognitum-gate-*`, `mcp-gate`, `mcp-brain` | No dedicated CI |

---

## 3. Quality Gate Analysis

### 3.1 Per-Workflow Quality Gates

| Workflow | Clippy/Lint | Tests | Formatting | Security Audit | Multi-platform | Coverage |
|----------|:-----------:|:-----:|:----------:|:--------------:|:--------------:|:--------:|
| `agentic-synth-ci.yml` | ESLint, TS typecheck | Unit, integration, CLI | -- | npm audit | 3 OS x 3 Node | codecov |
| `benchmarks.yml` | -- | Benchmarks only | -- | -- | Linux only | -- |
| `build-attention.yml` | -- | Module load test | -- | -- | 5 native + WASM | -- |
| `build-gnn.yml` | -- | -- | -- | -- | 7 native | -- |
| `build-graph-node.yml` | -- | npm test (native only) | -- | -- | 5 native | -- |
| `build-graph-transformer.yml` | -- | -- | -- | -- | 7 native + WASM | -- |
| `build-native.yml` | -- | Module load test | -- | -- | 5 native | -- |
| `build-router.yml` | -- | npm test (native only) | -- | -- | 5 native | -- |
| `build-rvf-node.yml` | -- | Module load test | -- | -- | 5 native | -- |
| `build-tiny-dancer.yml` | -- | npm test (native only) | -- | -- | 5 native | -- |
| `build-verified.yml` | Clippy (-D warnings) | cargo test | -- | -- | Linux only | -- |
| `docker-publish.yml` | -- | Extension load test | -- | -- | linux/amd64 + arm64 | -- |
| `hooks-ci.yml` | -- | Rust + npm CLI tests, PG schema | -- | -- | Linux only | -- |
| `postgres-extension-ci.yml` | Clippy (-D warnings) | pgrx test, all-features | cargo fmt --check | cargo audit (rustsec) | Ubuntu + macOS | -- |
| `publish-all.yml` | -- | ruvector-math + attention tests | -- | -- | 5 native + WASM | -- |
| `release.yml` | Clippy (-D warnings, workspace) | cargo test --workspace | cargo fmt --check | -- | 5 native + WASM | -- |
| `ruvector-postgres-ci.yml` | Clippy (-D warnings) | pgrx test, all-features, Docker | cargo fmt --check | cargo audit | Ubuntu + macOS | -- |
| `ruvllm-benchmarks.yml` | -- | Benchmarks only | -- | -- | macOS ARM64 + Linux | -- |
| `ruvllm-build.yml` | -- | Post-publish install test | -- | -- | 5 native | -- |
| `ruvllm-native.yml` | -- | -- | -- | -- | 5 native | -- |
| `ruvltra-tests.yml` | Clippy (-D warnings), RUSTDOCFLAGS | Unit, E2E, quant, thread safety | cargo fmt --check | -- | 3 OS (unit), macOS ARM64 | llvm-cov (60% threshold) |
| `sona-napi.yml` | -- | Post-publish install test | -- | -- | 7 native + universal macOS | -- |
| `thermorust-ci.yml` | Clippy (-D warnings) | cargo test (3 OS) | cargo fmt --check | -- | 3 OS | -- |
| `validate-lockfile.yml` | Lockfile validation | -- | -- | -- | Linux only | -- |
| `wasm-dedup-check.yml` | WASM dedup check | -- | -- | -- | Linux only | -- |

### 3.2 Summary of Quality Gate Coverage

| Quality Gate | Workflows That Implement It | Missing From |
|-------------|---------------------------|-------------|
| **Clippy/Lint** | `release.yml`, `postgres-extension-ci.yml`, `ruvector-postgres-ci.yml`, `build-verified.yml`, `ruvltra-tests.yml`, `thermorust-ci.yml`, `agentic-synth-ci.yml` (ESLint) | All `build-*` workflows, `benchmarks.yml`, `hooks-ci.yml`, `sona-napi.yml`, `publish-all.yml` |
| **Tests** | Most workflows have some form of testing | `build-gnn.yml`, `build-graph-transformer.yml`, `ruvllm-native.yml` have zero tests |
| **Formatting** | `release.yml`, `postgres-extension-ci.yml`, `ruvector-postgres-ci.yml`, `ruvltra-tests.yml`, `thermorust-ci.yml` | All `build-*` workflows, `agentic-synth-ci.yml`, `hooks-ci.yml`, `benchmarks.yml`, `sona-napi.yml` |
| **Security audit** | `postgres-extension-ci.yml`, `ruvector-postgres-ci.yml`, `agentic-synth-ci.yml` (npm audit) | No workspace-wide cargo audit. The release pipeline has no security gate. Most individual crate workflows skip it entirely |
| **Coverage** | `ruvltra-tests.yml` (llvm-cov, 60% threshold), `agentic-synth-ci.yml` (codecov) | All other workflows -- no coverage gating anywhere else |
| **Required checks** | None observed -- no branch protection rules requiring specific checks | All workflows lack required status check enforcement |

---

## 4. Workflow Health

### 4.1 Deprecated Actions

| Action | Used In | Issue | Recommended Replacement |
|--------|---------|-------|------------------------|
| `actions-rs/toolchain@v1` | `benchmarks.yml` (3 uses) | **Deprecated** -- `actions-rs` org is unmaintained since 2022 | `dtolnay/rust-toolchain@stable` (already used elsewhere) |
| `softprops/action-gh-release@v1` | `release.yml`, `publish-all.yml` | **v1 is outdated** -- `release-rvf-cli.yml` already uses `@v2` | Upgrade to `softprops/action-gh-release@v2` |

### 4.2 Action Version Pinning

**Most actions use major-version tags** (e.g., `@v4`, `@v3`), which is the standard practice. No SHA-pinning is used, which is less secure but typical for non-high-security repos.

| Action | Versions Used | Pinning Assessment |
|--------|--------------|-------------------|
| `actions/checkout` | `@v4` everywhere | Consistent |
| `actions/setup-node` | `@v4` everywhere | Consistent |
| `dtolnay/rust-toolchain` | `@stable` everywhere (except deprecated `actions-rs` uses) | Consistent |
| `actions/upload-artifact` | `@v4` everywhere | Consistent |
| `actions/download-artifact` | `@v4` everywhere | Consistent |
| `actions/cache` | `@v4` everywhere | Consistent |
| `Swatinem/rust-cache` | `@v2` everywhere | Consistent |
| `docker/build-push-action` | `@v5` everywhere | Consistent |
| `docker/setup-buildx-action` | `@v3` everywhere | Consistent |
| `codecov/codecov-action` | `@v4` | OK |
| `benchmark-action/github-action-benchmark` | `@v1` | Consistent |
| `peter-evans/dockerhub-description` | `@v4` | OK |

### 4.3 Redundant or Overlapping Workflows

| Overlap | Workflows | Assessment |
|---------|-----------|-----------|
| **ruvector-postgres CI** | `postgres-extension-ci.yml` AND `ruvector-postgres-ci.yml` | **Significant redundancy.** Both trigger on the same paths, both run lint+fmt+test+all-features+benchmark+security audit+packaging+Docker integration. The latter is more comprehensive (concurrency control, graph-complete feature). Recommend consolidating into one |
| **RuvLLM native builds** | `ruvllm-build.yml` AND `ruvllm-native.yml` | **Near-duplicate.** Both build ruvllm for 5 platforms and publish. `ruvllm-build.yml` uses Docker for Linux builds; `ruvllm-native.yml` uses napi-rs CLI. Recommend choosing one approach |
| **Attention builds** | `build-attention.yml` AND `publish-all.yml` | Partial overlap. `publish-all.yml` rebuilds attention native modules already handled by `build-attention.yml`. But `publish-all.yml` is release-only so this may be intentional |
| **Release pipelines** | `release.yml` AND `publish-all.yml` | Both trigger on `v*` tags. `release.yml` publishes crates to crates.io and creates a GitHub Release. `publish-all.yml` also publishes to crates.io, npm, and creates a GitHub Release. Running both on the same tag would cause conflicts |

### 4.4 Secrets Handling

All secrets are referenced via `${{ secrets.* }}` -- no hardcoded credentials found. Secrets used:

- `GITHUB_TOKEN` (automatic)
- `NPM_TOKEN`
- `CARGO_REGISTRY_TOKEN`
- `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`
- `GCP_SERVICE_ACCOUNT_KEY`, `GCP_PROJECT_ID`
- `GCP_WIF_PROVIDER`, `GCS_SERVICE_ACCOUNT`
- `PINATA_API_KEY`, `PINATA_SECRET`

### 4.5 Copilot Setup Workflow

`copilot-setup-steps.yml` has **malformed YAML indentation**. The file uses inconsistent indentation that would cause a parse error. This workflow is likely non-functional.

---

## 5. Gap Analysis

### 5.1 Critical Gaps

#### No Workspace-Wide CI on Push/PR

**The most significant gap.** There is no workflow that runs `cargo check --workspace`, `cargo test --workspace`, or `cargo clippy --workspace` on every push to `main` or on pull requests. The `release.yml` does this but only triggers on `v*` tags. This means:

- Changes to foundational crates like `ruvector-core` can be merged without any CI running
- Breakages across crate boundaries are not caught until release time
- Over 80 crates have zero CI on normal development workflows

**Recommendation**: Add a `ci.yml` workflow triggered on push/PR that runs at minimum:
```yaml
cargo check --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

#### No Workspace-Wide Security Audit

`cargo audit` is only run for `ruvector-postgres`. There is no periodic or PR-triggered security scan for the entire workspace dependency tree.

**Recommendation**: Add a scheduled (weekly) and PR-triggered `cargo audit` workflow for the entire workspace.

#### ruvector-core Has No Dedicated CI

As the foundational crate upon which everything depends, `ruvector-core` has no path-filtered CI. Any change to it could silently break downstream crates.

### 5.2 Missing Quality Gates

| Gap | Impact | Recommendation |
|-----|--------|---------------|
| No format check on most workflows | Inconsistent code style can be merged | Add `cargo fmt --check` to all Rust CI workflows |
| No clippy on build-* workflows | Lint violations merged via NAPI builds | Add clippy step to build workflows |
| No coverage thresholds (except ruvllm) | Coverage can regress without detection | Add llvm-cov or tarpaulin to core CI |
| No required status checks | PRs can be merged even if CI fails | Configure branch protection with required checks |
| No cargo deny or license check | Dependency license compliance unchecked | Add `cargo deny` for license and advisory checks |
| No MSRV check | `rust-version = "1.77"` in Cargo.toml but never validated in CI | Add MSRV testing with minimum supported Rust version |

### 5.3 Missing Platform Coverage

| Gap | Details |
|-----|---------|
| No Linux ARM64 test execution | ARM64 binaries are cross-compiled but never actually tested (cross-compilation only). Only native-platform tests run |
| No Windows ARM64 for most packages | Only `sona-napi.yml` builds for `aarch64-pc-windows-msvc` |
| No musl targets for most packages | Only `build-gnn.yml`, `build-graph-transformer.yml`, and `sona-napi.yml` build musl variants |
| No FreeBSD or other platform testing | Not critical but limits portability claims |

### 5.4 Missing WASM/Node.js Specific Testing

| Gap | Details |
|-----|---------|
| No browser-based WASM tests | WASM packages are built but never tested in a browser environment (e.g., via playwright/puppeteer) |
| No Node.js integration tests for WASM packages | `wasm-pack build --target nodejs` output is not `require()`-tested for most packages |
| No WASM size regression tracking | No workflow tracks the size of `.wasm` binaries over time |
| No `wasm-opt` optimization step | WASM binaries may not be optimally sized |

### 5.5 Missing Per-Crate CI for High-Value Crates

The following crates are significant but have no path-triggered CI:

| Crate | Risk | Reason |
|-------|------|--------|
| `ruvector-core` | **CRITICAL** | Foundation of all other crates |
| `ruvector-cnn` + `ruvector-cnn-wasm` | HIGH | ML inference crate with no tests in CI |
| `ruvector-mincut` family (5 crates) | MEDIUM | Graph partitioning algorithms |
| `ruvector-delta-*` (5 crates) | MEDIUM | Delta/incremental index crates |
| `rvAgent/*` (9 crates) | HIGH | Entire AI agent framework with no CI |
| `ruvix/*` (15 crates) | HIGH | Entire cognition kernel with no CI |
| `neural-trader-*` (4 crates) | MEDIUM | Trading/financial ML crates |

### 5.6 Workflow Configuration Issues

| Issue | Workflow | Details |
|-------|----------|---------|
| Malformed YAML | `copilot-setup-steps.yml` | Incorrect indentation will cause parse failures |
| Hardcoded version | `build-tiny-dancer.yml` | Line 158: `VERSION="0.1.15"` is hardcoded instead of reading from `package.json` |
| Suppressed failures | Multiple workflows | Many `\|\| echo "..."` and `continue-on-error: true` patterns hide real failures |
| Missing concurrency control | Most workflows | Only `ruvector-postgres-ci.yml` uses `concurrency:` to cancel in-progress runs. Other workflows can pile up |
| No timeout | Most workflows | Only `benchmarks.yml`, `ruvllm-benchmarks.yml`, and `ruvltra-tests.yml` set `timeout-minutes`. Others could run indefinitely |

---

## 6. Recommendations (Prioritized)

### P0 -- Must Fix

1. **Add a workspace-wide CI workflow** triggered on push to main and all PRs: `cargo check --workspace`, `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --all -- --check`
2. **Fix the `copilot-setup-steps.yml` YAML indentation** -- currently unparseable
3. **Replace deprecated `actions-rs/toolchain@v1`** with `dtolnay/rust-toolchain@stable` in `benchmarks.yml`
4. **Resolve the dual postgres CI** -- consolidate `postgres-extension-ci.yml` and `ruvector-postgres-ci.yml` into one workflow
5. **Resolve the dual release pipeline** -- `release.yml` and `publish-all.yml` both trigger on `v*` tags and both attempt to publish to crates.io and create GitHub Releases

### P1 -- Should Fix

6. **Add workspace-wide `cargo audit`** (scheduled weekly + on PRs)
7. **Add concurrency control** to all long-running workflows to prevent queue pile-up
8. **Add `timeout-minutes`** to all jobs (30 min for builds, 60 min for benchmarks)
9. **Upgrade `softprops/action-gh-release@v1`** to `@v2` in `release.yml` and `publish-all.yml`
10. **Remove hardcoded version** in `build-tiny-dancer.yml` publish step
11. **Add dedicated CI for `ruvector-core`** with path-filtered triggers
12. **Resolve duplicate ruvllm build workflows** (`ruvllm-build.yml` vs `ruvllm-native.yml`)

### P2 -- Nice to Have

13. **Add MSRV testing** with `rust-version = "1.77"` validation
14. **Add `cargo deny`** for license compliance and duplicate dependency detection
15. **Add WASM size tracking** to monitor binary sizes across releases
16. **Add browser-based WASM testing** for key WASM packages
17. **Add test coverage** to the workspace-wide CI using `cargo llvm-cov`
18. **Configure branch protection** with required status checks on `main`
19. **Add CI for `rvAgent/` and `ruvix/`** sub-workspaces -- currently 24 crates with zero coverage

---

## Appendix: Full Crate-to-Workflow Matrix

The following table maps every workspace member to its CI coverage level.

| Coverage Level | Count | Description |
|---------------|-------|-------------|
| FULL | 7 | Dedicated workflow with lint + tests + format + audit |
| BUILD-ONLY | 10 | Dedicated build workflow but no lint/test/format gates |
| WORKSPACE-ONLY | ~80 | Only tested in `release.yml` workspace build (tag-triggered) |
| NONE | ~10 | Excluded from workspace and no dedicated workflow |

**Key**: FULL coverage applies to: `ruvector-postgres`, `ruvector-verified`, `thermorust`, `ruvllm`, `ruvector-cli` (hooks), and the `agentic-synth` npm package.
