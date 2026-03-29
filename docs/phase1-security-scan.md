# Phase 1 Security Scan Report

**Date**: 2026-03-29
**Scanner**: QE Security Scanner v3 (SAST + Secrets)
**Branch**: qe-working-branch
**Scope**: Steps 1.6, 1.7, 1.8 of Phase 1 automated quality analysis

---

## Step 1.6: Unsafe Block Classification

### Summary

| Metric | Value |
|--------|-------|
| Total files with `unsafe` | **239** |
| Total `unsafe` keyword references | **2,129** |
| Domains with unsafe code | 7 of 10 |
| Highest-density domain | D10 Specialized/Other (1,065 refs across 123 files) |
| Highest-density crate domain | D5 Neural/ML (489 refs across 69 files) |

### Top 20 Files by Unsafe Count

| Rank | File | Count | Domain |
|------|------|------:|--------|
| 1 | `scripts/patches/hnsw_rs/src/libext.rs` | 92 | D10 (HNSW patch) |
| 2 | `patches/hnsw_rs/src/libext.rs` | 92 | D10 (HNSW patch) |
| 3 | `crates/ruvector-postgres/src/distance/simd.rs` | 78 | D4 Security & Persistence |
| 4 | `crates/micro-hnsw-wasm/src/lib.rs` | 55 | D6 WASM Bindings |
| 5 | `crates/rvf/rvf-wasm/src/lib.rs` | 47 | D6 WASM Bindings |
| 6 | `crates/ruvix/benches/src/linux.rs` | 42 | D10 (RuVix OS kernel) |
| 7 | `crates/ruvector-core/src/simd_intrinsics.rs` | 42 | D1 Core Vector DB |
| 8 | `crates/ruvector-postgres/src/index/hnsw_am.rs` | 40 | D4 Security & Persistence |
| 9 | `crates/ruvector-postgres/src/index/ivfflat_am.rs` | 29 | D4 Security & Persistence |
| 10 | `crates/ruvix/crates/aarch64/src/registers.rs` | 27 | D10 (RuVix OS kernel) |
| 11 | `examples/edge-net/src/compute/simd.rs` | 26 | D10 (examples) |
| 12 | `crates/ruvllm/src/memory_pool.rs` | 26 | D5 Neural/ML |
| 13 | `crates/ruvector-postgres/src/types/halfvec.rs` | 26 | D4 Security & Persistence |
| 14 | `crates/cognitum-gate-kernel/src/lib.rs` | 26 | D10 (cognitum) |
| 15 | `examples/ruvLLM/src/simd_inference.rs` | 25 | D10 (examples) |
| 16 | `crates/cognitum-gate-kernel/src/shard.rs` | 25 | D10 (cognitum) |
| 17 | `examples/wasm/ios/src/lib.rs` | 23 | D10 (examples) |
| 18 | `crates/ruvector-core/src/arena.rs` | 23 | D1 Core Vector DB |
| 19 | `crates/ruvix/crates/smp/src/barriers.rs` | 22 | D10 (RuVix OS kernel) |
| 20 | `crates/ruvector-cnn/src/simd/avx2.rs` | 22 | D5 Neural/ML |

### Unsafe Count by Domain

| Domain | Files | Unsafe Refs | % of Total | Risk Notes |
|--------|------:|------------:|-----------:|------------|
| **D1** Core Vector DB | 5 | 95 | 4.5% | SIMD intrinsics + arena allocator; justified for performance |
| **D2** Graph Database | 5 | 14 | 0.7% | Low unsafe density; mostly optimization |
| **D3** Distributed Systems | 0 | 0 | 0.0% | No unsafe code -- good |
| **D4** Security & Persistence | 21 | 265 | 12.4% | Postgres FFI + SIMD distance; requires audit of pgrx safety |
| **D5** Neural/ML | 69 | 489 | 23.0% | Largest crate domain; SIMD kernels, Metal FFI, memory pools |
| **D6** WASM Bindings | 16 | 200 | 9.4% | WASM allocator setup + SIMD operations |
| **D7** Node.js Bindings | 1 | 1 | 0.0% | Minimal -- 1 reference only |
| **D8** CLI & Router | 0 | 0 | 0.0% | No unsafe code -- good |
| **D9** UI Layer | 0 | 0 | 0.0% | No unsafe code -- good |
| **D10** Specialized/Other | 123 | 1,065 | 50.0% | See breakdown below |

### D10 Breakdown (Specialized/Other)

| Sub-category | Unsafe Refs | Files | Notes |
|--------------|------------:|------:|-------|
| RuVix OS kernel (`crates/ruvix/`) | 421 | 53 | Bare-metal aarch64 OS -- unsafe is inherent to hardware access |
| Examples (`examples/`) | 227 | ~20 | Research/demo code; not shipped in production |
| HNSW patches (`patches/`, `scripts/patches/`) | 196 | 6 | Vendored upstream code; duplication between patches/ and scripts/patches/ |
| Cognitum gate kernel | 76 | 4 | Verifiable compute kernel |
| Other misc (rvf non-wasm, rvlite, rvAgent, etc.) | 145 | ~40 | Mixed FFI, crypto, solver code |

### Key Observations

1. **SIMD dominates unsafe usage.** The majority of unsafe blocks are in SIMD intrinsic wrappers (`simd_intrinsics.rs`, `avx2.rs`, `neon.rs`, `wasm.rs`). This is expected and standard practice in Rust for performance-critical vector operations.

2. **Postgres FFI is the second major source.** The `ruvector-postgres` crate has 21 files / 265 unsafe references, driven by pgrx (Postgres extension framework) which requires unsafe for C-level FFI. The `distance/simd.rs` file alone has 78 references.

3. **RuVix OS kernel is expected.** The 53 files with 421 unsafe refs in `crates/ruvix/` are a bare-metal aarch64 OS kernel. Unsafe is inherently required for MMU, interrupt handlers, MMIO registers, and boot sequences.

4. **Duplicate HNSW patches.** The files in `patches/hnsw_rs/` and `scripts/patches/hnsw_rs/` appear to be exact duplicates (same counts: 92, 4, 2). One copy should be eliminated.

5. **No unsafe in D3, D8, D9.** Distributed systems, CLI, and UI have zero unsafe code, which is the correct design.

---

## Step 1.7: Secrets Scan

### Scan Methodology

Patterns searched across all `.rs`, `.ts`, `.js`, `.json`, `.yaml`, `.yml`, `.toml` files:
- API key formats: `sk-*`, `api_key=`, `apiKey:`, `API_KEY=`, `Bearer` tokens
- Token/password assignments: `token = "..."`, `secret = "..."`, `password = "..."`
- Private keys: `-----BEGIN ... PRIVATE KEY-----`
- Credential URLs: `://user:pass@host`
- Platform tokens: `xoxb-*`, `xapp-*`, `ghp_*`, `gho_*`, `glpat-*`

### Findings Summary

| Severity | Count | Category |
|----------|------:|----------|
| CRITICAL (real secrets) | **0** | -- |
| HIGH (hardcoded credentials) | **0** | -- |
| MEDIUM (credential URLs in config) | **3** | Docker/dev database URLs |
| LOW (false positives / test data) | **~45** | Placeholder values, test fixtures, PII detection tests |
| INFO (example/documentation patterns) | **~15** | CLI help text showing export commands |

### MEDIUM Findings: Database Credential URLs in Docker/Config

These are development/CI-only database URLs with default credentials. They are not production secrets, but they should be reviewed:

| # | File | Line | Value | Assessment |
|---|------|------|-------|------------|
| M1 | `npm/packages/ruvbot/docker-compose.yml` | 30 | `postgresql://ruvbot:ruvbot_dev@postgres:5432/ruvbot` | Dev-only Docker Compose; acceptable |
| M2 | `crates/ruvector-postgres/docker/docker-compose.yml` | 21 | `postgres://ruvector:ruvector@postgres:5432/ruvector_test` | Test-only Docker Compose; acceptable |
| M3 | `crates/ruvector-postgres/docker/docker-compose.integration.yml` | 56,87 | `postgres://ruvector:ruvector@postgres:5432/ruvector_test` | Integration test Docker; acceptable |

**Assessment**: All three are Docker Compose files for local development and CI. The passwords are default dev values (`ruvbot_dev`, `ruvector`). These are NOT production credentials. **False positive for production risk, but recommend using environment variable substitution** (`${DB_PASSWORD}`) instead of inline credentials, as a defense-in-depth measure.

### Additional Database URL Occurrences (INFO level)

| File | Context | Assessment |
|------|---------|------------|
| `npm/packages/postgres-cli/src/commands/install.ts:819` | Default connection string for fresh install | Template value |
| `ui/ruvocal/src/lib/server/database/postgres.ts:24` | Fallback URL for local dev | Non-production default |
| `.github/workflows/hooks-ci.yml:135,153,157` | CI workflow database setup | CI-only |
| `crates/ruvector-cli/src/cli/hooks_postgres.rs:392,410` | Unit test fixtures | Test data |

### False Positives (Correctly Excluded)

The following matched patterns but are NOT real secrets:

1. **PII detection test data** (7 occurrences): Files like `crates/rvf/rvf-federation/src/pii_strip.rs`, `crates/mcp-brain-server/src/verify.rs`, and `crates/mcp-brain/src/pipeline.rs` contain fake API keys and tokens as test inputs to verify that the PII stripping / redaction logic works correctly. These are explicitly designed to test secret detection.

2. **Placeholder/example API keys** (~12 occurrences): Strings like `"YOUR_API_KEY"`, `"your-api-key"`, `"test-key-123"`, `"test-api-key"` in ruvbot, agentic-synth, and scipix packages. These are configuration templates or test fixtures.

3. **CLI help text** (~8 occurrences): Messages like `export SLACK_BOT_TOKEN="xoxb-your-bot-token"` in `channels.ts/.js` are user-facing instructions, not actual tokens.

4. **Security eval templates** (4 occurrences): Files in `.claude/skills/*/evals/` contain deliberately insecure code snippets used to test the security scanner itself.

5. **Test setup files** (4 occurrences): `ruvbot/tests/setup.ts` sets dummy env vars like `xoxb-test-token` for test isolation.

6. **Password in example code** (1 occurrence): `examples/edge-net/src/identity/mod.rs:340` has `"secure_password_123"` in a test function, and `examples/edge-net/pkg/join.js:640` has a default password with a comment warning to use a strong password in production.

### Private Key References

All 7 matches for `-----BEGIN` are in:
- PII stripping / redaction code and tests (verify patterns are detected and redacted)
- A key-format detection check in `ruvix/crates/cli/src/commands/keys.rs`

**No actual private key material was found in the repository.**

### Verdict: PASS

**No real secrets, API keys, tokens, or private keys were found committed to the repository.** All matches are either test fixtures, placeholder values, documentation examples, or security detection test data.

---

## Step 1.8: .env File Audit

### Findings

| Check | Result |
|-------|--------|
| File exists | YES: `ui/ruvocal/.env` |
| Tracked in git | **YES** -- `git ls-files` confirms it is tracked |
| Contains real secrets | **NO** -- all sensitive fields are empty or commented out |
| Root `.gitignore` coverage | `.env` and `**/.env` rules exist (lines 42-45) |
| Local `.gitignore` override | `ui/ruvocal/.gitignore` line 13: `!.env` (negation -- re-includes the file) |

### .env File Content Analysis

The file at `ui/ruvocal/.env` is a **configuration template** (not a secret store). Key observations:

1. **Header clearly states**: "Use .env.local to change these variables / DO NOT EDIT THIS FILE WITH SENSITIVE DATA"

2. **All sensitive fields are empty or commented**:
   - `OPENAI_API_KEY=#your provider API key...` (commented placeholder)
   - `DATABASE_URL=#postgresql://ruvocal:password@localhost:5432/ruvocal` (commented out)
   - `OPENID_CLIENT_SECRET=` (empty)
   - `ADMIN_API_SECRET=# secret to admin API calls...` (commented placeholder)
   - All `LLM_ROUTER_*` config values are empty

3. **Non-sensitive defaults present**:
   - `OPENAI_BASE_URL=https://router.huggingface.co/v1` (public API endpoint)
   - `PUBLIC_APP_NAME=ChatUI`
   - `PUBLIC_APP_DESCRIPTION="Making the community's best AI chat models available to everyone."`
   - Various feature flags (all empty/false)

### Gitignore Rule Chain

The root `.gitignore` at lines 42-45 contains:
```
.env
**/.env
**/.env.local
**/.env.*.local
```

However, `ui/ruvocal/.gitignore` at line 13 contains:
```
!.env
```

This negation pattern **intentionally re-includes** the `.env` file for the ruvocal UI, which is the standard pattern for SvelteKit/chat-ui projects. The `.env` file serves as a **documented configuration template** that ships with the project, while `.env.local` (which IS gitignored) holds actual secrets.

### Verdict: FALSE POSITIVE

The tracked `.env` file is a **configuration template by design**. It contains zero real credentials. The `.env.local` override (where real secrets go) is properly gitignored. This follows the standard SvelteKit convention where `.env` is the default/template and `.env.local` holds overrides.

**Recommendation**: Add a comment at the top of `ui/ruvocal/.gitignore` explaining why `!.env` is present, for future auditors:
```
# .env is intentionally tracked as a config template (no secrets).
# Real secrets go in .env.local which IS gitignored.
!.env
```

---

## Overall Security Scan Summary

| Step | Status | Findings |
|------|--------|----------|
| 1.6 Unsafe Blocks | INFORMATIONAL | 239 files, 2,129 refs. Dominated by SIMD (expected), Postgres FFI, and OS kernel code. No anomalies. |
| 1.7 Secrets Scan | **PASS** | 0 real secrets found. 3 medium-severity dev DB URLs in Docker configs (acceptable). |
| 1.8 .env Audit | **PASS (False Positive)** | Tracked `.env` is a config template with no secrets. `.env.local` is properly gitignored. |

### Recommendations

1. **Deduplicate HNSW patches**: `patches/hnsw_rs/` and `scripts/patches/hnsw_rs/` contain identical files (libext.rs: 92 unsafe refs each). Remove one copy.

2. **Docker Compose credential hygiene**: Replace inline database passwords in Docker Compose files with `${DB_PASSWORD:-default}` environment variable substitution for defense-in-depth.

3. **Unsafe audit priority**: Focus safety audits on:
   - `ruvector-postgres/src/distance/simd.rs` (78 unsafe, highest in production code)
   - `ruvector-postgres/src/index/hnsw_am.rs` (40 unsafe, index access methods)
   - `ruvector-core/src/simd_intrinsics.rs` (42 unsafe, core vector operations)
   - `ruvllm/src/memory_pool.rs` (26 unsafe, memory management)

4. **Document .gitignore intent**: Add explanatory comment in `ui/ruvocal/.gitignore` for the `!.env` negation rule.

---

*Report generated by QE Security Scanner v3 -- Phase 1 Automated Analysis*
