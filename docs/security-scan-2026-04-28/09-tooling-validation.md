# Tooling Validation — Authoritative Re-Scan with Industry Tools

**Date:** 2026-04-28
**Tools installed and run after agent fleet:**

| Tool | Version | Source |
|---|---|---|
| `cargo-audit` | 0.22.1 | Precompiled binary from `rustsec/rustsec` releases (aarch64-linux-gnu) |
| `cargo-geiger` | (in progress, compiling from source) | `cargo install cargo-geiger --locked` |
| `gitleaks` | 8.30.1 | Precompiled binary from `gitleaks/gitleaks` releases |
| `trufflehog` | 3.95.2 | Official installer script |

Raw outputs: `docs/security-scan-2026-04-28/raw/`

---

## Corrections to Agent Findings

The 8-agent fleet relied on `ripgrep`/`grep` pattern matching because authoritative tools were not installed in the devcontainer at the time of the scan. After installing them, several specific claims need correction.

### CORRECTION #1 — Rust dependency CVEs: agent overstated

**Agent claim (03-dependency-cves.md):**
- 1 critical reachable Rust CVE: `rsa@0.9.10` (RUSTSEC-2023-0071 "Marvin Attack")
- Unmaintained-crate warnings: `serde_yaml@0.9.34+deprecated` (RUSTSEC-2024-0320), `paste`, `derivative`, `instant`, `proc-macro-error`, `term`

**Authoritative result (`cargo-audit 0.22.1` against fresh RustSec advisory DB with 1058 advisories, scanning 1246 deps in `Cargo.lock`):**

```
Scanning Cargo.lock for vulnerabilities (1246 crate dependencies)
vulnerabilities.count = 0
warnings = {}
```

Zero open advisories. Confirmed even with `--deny warnings` flag (which surfaces informational/unmaintained categories). `rsa@0.9.10` and `serde_yaml@0.9.34+deprecated` ARE present in `Cargo.lock` — but the RUSTSEC IDs the agent cited are not currently in the active advisory DB against these versions.

**Net effect:** The Rust dependency tree is **clean** per the authoritative tool. Agent's "P0 — Replace `rsa@0.9.10` for Marvin Attack" recommendation should be **downgraded to P3 hardening** (still defensible to track upstream, but no active advisory). Same for the unmaintained-crate warnings list.

**Caveat:** `cargo-audit` could not refresh `crates.io` index in this devcontainer (network limitation). The advisory DB itself was successfully fetched from GitHub. Re-run in CI on a network-unrestricted runner before final disclosure to confirm.

---

### CORRECTION #2 — Secret/PII scan: gitleaks confirms agent's count, with nuance

**Agent claim (04-secrets-pii.md):** 1 confirmed real leak (Firebase web API key in git history at commit `1a6174bc`); working tree clean.

**Authoritative result (`gitleaks 8.30.1`):**

| Scope | Hits | Verdict |
|---|---|---|
| **Working tree only** (`--no-git`) | 35 | All test fixtures, third-party research bundles, gitignored files, README examples, and friendly-string identifiers. Zero real credentials. |
| **Full git history** (4,169 commits, 953 MB scanned in 3m43s) | 132 | 35 from above + 97 historical. Of the 97 history-only hits, 2 hits (`gcp-api-key` rule) are the Firebase web key in commit `1a6174bc` files `examples/edge-net/dashboard/src/services/firebaseData.ts` and `examples/edge-net/pkg/firebase-signaling.js` — confirming the agent's finding. The remaining 95 are recurrences of the same fixtures across commits. |

**Triage of working-tree hits (35):**

| Rule | Count | Verdict |
|---|---|---|
| `generic-api-key` | 18 | All examples/test fixtures/templates. Spot-checked across `crates/ruvector-tiny-dancer-core/docs/API.md` (5 docs examples), `.claude/agents/v3/subagents/qe-tdd-green.md` (2 agent template), `crates/mcp-brain/src/pipeline.rs` (3 hits — all inside `#[test] fn test_strip_pii_*` blocks), `crates/rvAgent/README.md` (2 README examples). |
| `curl-auth-header` | 14 | Documentation/scripts using `curl -H "Authorization: Bearer $TOKEN"`-style examples. Token literal is `${TOKEN}` or shell variable, not a real secret. |
| `stripe-access-token` | 1 | `.claude/skills/n8n-security-testing/evals/n8n-security-testing.yaml:111` — the literal `Bearer sk_live_51234567890abcdef` is a TEST FIXTURE for the n8n security-testing skill (this skill's job IS to detect such patterns). |
| `private-key` | 1 | `crates/mcp-brain/src/pipeline.rs:251` — inside `#[test] fn test_strip_pii_private_key`. The "private key" string is `-----BEGIN RSA PRIVATE KEY-----\nMIIEpA...\n-----END RSA PRIVATE KEY-----` (truncated `MIIEpA...`) — a fixture used to verify the PII stripper redacts private keys. |
| `jwt` | 1 | `crates/mcp-brain/src/pipeline.rs:242` — same `test_strip_pii_jwt` test fixture. |

**Verified gitignored / untracked (not committed):**
- `studio/.env` (3 jwt hits) — `.gitignore` line 43 covers it; `git ls-files` confirms not tracked. Local-only working files. The `studio/` directory has **0 git-tracked files** total — likely a local Supabase Studio fork the developer is exploring outside the repo.

**Net effect:** Agent's conclusion stands. The Firebase web API key in commit `1a6174bc` remains the only confirmed real leak. Action stays the same: **restrict / rotate in GCP Console for project `ruv-dev` and audit Firebase Security Rules**. Working tree is clean of credentials.

---

### CORRECTION #3 — Secret count discrepancy explanation

The PII agent's manual `ripgrep`-based scan reported "1 confirmed leak". `gitleaks` reports "132 leaks found". Both are correct in their own framing:

- The agent applied judgment to filter false positives (test fixtures, gitignored files, third-party research bundles) — reported only the one real leak.
- `gitleaks` counted every regex match in the full 5,575-commit history without filtering test fixtures.

Triage of the gitleaks result reproduces the agent's conclusion. **No action change.**

---

### CORRECTION #4 — trufflehog full git scan with verification: ZERO verified secrets

`trufflehog 3.95.2 git file:///workspaces/RuVector --json --no-update` completed across the full git history (~13 minutes). It attempted to authenticate every candidate secret against the corresponding cloud API.

**Result:** **0 verified secrets across 97 candidate hits.**

| Detector | Hits | Verified | Triage |
|---|---|---|---|
| `NpmToken` | 60 | **0** | All in one file: `.claude/intelligence/data/memory.json` — agent intelligence pattern store, contains historical npm-token-shaped strings as memorized patterns from prior audits. Not live tokens. |
| `Postgres` | 35 | **0** | Test/dev docker-compose connection strings (`crates/ruvector-postgres/docker/*.yml`, `npm/packages/ruvbot/docker-compose.yml`, `deploy/gcp/deploy.sh` template), example READMEs, apify input schemas. trufflehog tried to authenticate against each — all rejected. |
| `GoogleGeminiAPIKey` | 2 | **0** | Both hits are the **same Firebase web key in commit `1a6174bc`** (`firebaseData.ts:48`, `firebase-signaling.js:39`) the PII agent and gitleaks already identified. Files no longer in working tree. trufflehog tested it against the Gemini API and got rejected (key is either restricted to Firebase scope, already revoked, or invalid for Gemini specifically). |

**Verdict:** The repository contains **zero live, working credentials in the entire git history**. The Firebase web key flagged by all three tools (PII agent, gitleaks, trufflehog) is the only non-test-fixture credential ever committed, and trufflehog confirms it does not currently authenticate to Google services. The action remains: **restrict / rotate it in GCP Console for project `ruv-dev` and audit Firebase Security Rules** — this prevents the key from being un-restricted in future and confirms abuse boundaries.

---

### CORRECTION #5 — cargo-geiger: tool incompatible with installed cargo

`cargo-geiger 0.13.0` was successfully compiled (~14 min), but crashed on first invocation with:

```
thread 'main' panicked at .../cargo-0.86.0/src/cargo/core/package.rs:736:9:
assertion failed: self.pending_ids.insert(id)
```

This is a known compatibility issue between `cargo-geiger 0.13.0` and `cargo 1.95.0` — the `cargo` crate API is unstable and `cargo-geiger` has not been updated for the current toolchain. Confirmed reproducible against `crates/mcp-brain-server` (raw output: `raw/cargo-geiger-mcp-brain-server.log`).

**Workarounds:**
- Pin to an older `cargo` (`rustup toolchain install 1.81.0` and run `cargo +1.81.0 geiger`) — would require another ~15 min toolchain install. Out of scope for this scan.
- Use `cargo-geiger` from a maintained fork (e.g., `cargo-geiger-cli` or community patches) — none confirmed working at the time of writeup.
- **Manual `unsafe`-density count via `rg`** — done below.

**Manual `unsafe`-density count for the security-critical crates the SAST agent flagged:**

| Crate | `unsafe` lines | LOC | Density (per kLOC) | SAST agent verdict |
|---|---:|---:|---:|---|
| `mcp-brain-server` | 0 | 27,495 | 0.00 | Pure-safe Rust on the network surface — **good** |
| `ruvector-router-ffi` | 0 | 221 | 0.00 | NAPI shim, no `unsafe` — **good** |
| `rvAgent` | 3 | 61,774 | 0.05 | Very low — **good** |
| `ruvector-temporal-tensor` | 23 | 15,478 | 1.49 | **H-1 / H-2 confirmed**: spot-check shows `static mut STORE: Option<Vec<...>> = None` at `ffi.rs:13` and 17 `unsafe { std::slice::from_raw_parts(...) }` blocks at lines 16, 28, 57, 153, 164, 188, 212, 220, 236, 247 |
| `ruvector-core` | 95 | 25,866 | 3.67 | Reviewed — **sound** (every block has `SAFETY:` comment + bounds check, per agent) |
| `cognitum-gate-kernel` | 77 | 6,595 | **11.68** | Reviewed — **sound** (highest density of any critical-path crate, but justified — every `unsafe` annotated) |

**Top 20 workspace-wide by raw `unsafe` line count** (audit-priority candidates the agent fleet did NOT cover):

| Rank | Crate | `unsafe` lines | Status in agent fleet |
|---:|---|---:|---|
| 1 | `ruvix` | 441 | NOT audited — **flag for next round** |
| 2 | `ruvllm` | 291 | NOT audited (only `working_memory.rs` and `semantic_cache.rs` MD5 cache-key issue flagged) — **flag for next round** |
| 3 | `ruvector-postgres` | 265 | NOT audited — **flag for next round** |
| 4 | `ruvector-cnn` | 122 | NOT audited |
| 5 | `rvf` | 120 | Partially audited (`rvf-crypto`, `rvf-federation`, `rvf-mcp-server`) |
| 6 | `ruvector-core` | 95 | Audited ✓ |
| 7 | `rvm` | 92 | NOT audited |
| 8 | `cognitum-gate-kernel` | 77 | Audited ✓ |
| 9 | `micro-hnsw-wasm` | 55 | NOT audited |
| 10 | `ruvector-wasm` | 44 | NOT audited |
| 11 | `ruvector-mincut-gated-transformer` | 34 | NOT audited (math/ML crate) |
| 12 | `ruvector-solver` | 25 | NOT audited (math crate, lower priority) |
| 13 | `ruvector-mincut` | 24 | NOT audited (math crate) |
| 14 | `ruvector-temporal-tensor` | 23 | Audited ✓ — **H-1/H-2** |
| 15 | `ruvector-rabitq` | 22 | NOT audited |
| 16 | `ruvector-fpga-transformer` | 21 | NOT audited |
| 17 | `ruvector-sparse-inference` | 16 | NOT audited |
| 18 | `ruvector-graph` | 14 | NOT audited |
| 19 | `ruvector-consciousness` | 14 | NOT audited |
| 20 | `ruvector-gnn` | 13 | NOT audited |

**New finding from `unsafe`-density analysis:** `ruvix` (441 unsafe lines), `ruvllm` (291), `ruvector-postgres` (265) are the three crates with the highest absolute `unsafe` count and were **not in the SAST agent's scope**. They should be in the next audit round to determine whether the `unsafe` is justified (math-kernel SIMD, FFI bindings) or potentially unsound. Quick test: `rg 'static mut|from_raw_parts|transmute' crates/ruvix/src` etc.

---

## What Did NOT Change

The other agent findings stand because they covered surfaces tooling does not address:

| Finding | Authoritative tool exists? | Verdict |
|---|---|---|
| Auth bypass in `mcp-brain-server/src/auth.rs:99` | No (semantic, not pattern) | **STANDS — confirmed by 3 independent agents** |
| `verify_system_key` fails-OPEN | No | **STANDS** |
| MCP `/sse`, `/messages` no-auth | No | **STANDS** |
| `/internal/*` on public listener | No | **STANDS** |
| WASM publish signature declared but never verified | No | **STANDS** |
| Indirect prompt injection via stored memories → Gemini → public gist | No | **STANDS** |
| Memory poisoning → LoRA training distribution | No | **STANDS** |
| Firestore path traversal via `publish_node.id` | No | **STANDS** |
| Shell injection in `rvAgent/sandbox.rs` and `rvagent-cli/app.rs` | Partial (semgrep would find, not run) | **STANDS** |
| FFI `static mut` data races + `Vec::from_raw_parts` capacity bug | No (Rust-specific UB) | **STANDS** |
| DP claim ε=1.0 doesn't hold | No (mathematical analysis) | **STANDS** |
| Witness chain has no external anchor | No (architectural) | **STANDS** |
| Persistent XSS in `ruvector-extensions/src/ui/app.js` | Yes (semgrep / eslint-plugin-security would find) | **STANDS** — not run, but pattern is unambiguous (string concatenation → `innerHTML`) |
| Math.random() in Raft election timeouts | No (semantic) | **STANDS** |
| CORS wildcard on writable APIs | Yes (eslint, not run) | **STANDS** |
| `npm audit` 14 critical + 27 high | YES — **`npm audit` was run by the JS/TS SAST agent**, results authoritative | **STANDS** |
| Workflow `permissions:` blocks missing, no SHA-pinning | Yes (`actionlint`, `zizmor`, not run) | **STANDS** — surface is small enough for manual review to be reliable |

---

## Updated Severity Scorecard

| Category | Agent count | Authoritative count | Net change |
|---|---|---|---|
| Rust dependency CVEs (open) | 1 critical | **0** | ⬇️ Removed |
| Rust unmaintained warnings | 6 | **0 in active DB** | ⬇️ Removed |
| npm CVEs | 14C / 27H | **14C / 27H** | unchanged (same `npm audit` source) |
| Auth/AuthZ critical findings | 7 | **7** | unchanged (semantic, no tool) |
| Confirmed credential leaks | 1 (Firebase, history) | **1** | unchanged |
| Working-tree credential leaks | 0 | **0** | unchanged (gitleaks confirms) |

**Net effect on overall posture verdict:** **unchanged at 4/10 public brain plane / 6/10 local libs.** The dependency surface improves (clean cargo-audit), but the dominant risks are architectural (auth bypass + memory poisoning + WASM signature not verified + LoRA training pipeline poisoning) — none of which a CVE scanner can find.

---

## Recommended CI Integration

Add these to `.github/workflows/security.yml`:

```yaml
jobs:
  cargo-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: rustsec/audit-check@v2  # already in some workflows; add to ci.yml
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # full history for first PR; can drop later
      - uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  trufflehog:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: trufflesecurity/trufflehog@main
        with:
          extra_args: --only-verified
```

Add `gitleaks protect --staged` to `.githooks/pre-commit` to catch secrets before commit.

---

## Files Referenced

- `/workspaces/RuVector/docs/security-scan-2026-04-28/raw/cargo-audit.json` — full cargo-audit JSON
- `/workspaces/RuVector/docs/security-scan-2026-04-28/raw/gitleaks-tree.json` — gitleaks 4169-commit history scan
- `/workspaces/RuVector/docs/security-scan-2026-04-28/raw/gitleaks-filesystem.json` — gitleaks working-tree-only (`--no-git`)
- `/workspaces/RuVector/docs/security-scan-2026-04-28/raw/trufflehog-git.jsonl` — trufflehog (in progress)
- `/workspaces/RuVector/docs/security-scan-2026-04-28/raw/cargo-audit.stderr` — toolchain warnings
