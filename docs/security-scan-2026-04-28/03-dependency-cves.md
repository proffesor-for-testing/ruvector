# Dependency Vulnerability Audit

**Date:** 2026-04-28
**Scope:** Root `Cargo.lock` (1,246 crates), `npm/package-lock.json` (2,082 deps), root `package.json`, `ui/ruvocal/package.json`, spot-check of `mcp-brain-server` / `mcp-gate` / `ruvector-router-ffi` Cargo.toml files.
**Auditor:** security-auditor (V3, ReasoningBank-augmented), read-only.

---

## Tooling output summary

| Tool | Status | Result |
|---|---|---|
| `cargo audit` | NOT RUN | `cargo` toolchain not installed in this devcontainer. Manual analysis of `Cargo.lock` performed against the RUSTSEC advisory DB (knowledge cutoff Jan 2026). |
| `npm audit` (`/workspaces/RuVector`) | SKIPPED | No `package-lock.json` at repo root and policy forbids generating one (read-only audit). |
| `npm audit` (`/workspaces/RuVector/npm`) | OK | **68 vulns: 14 critical / 27 high / 24 moderate / 3 low** across 2,082 deps. |
| `npm audit` (`/workspaces/RuVector/ui/ruvocal`) | SKIPPED | No lockfile present — cannot audit without generating one. **Recommend committing `package-lock.json` for ui/ruvocal.** |

**Gap:** `ui/ruvocal` is a SvelteKit chat UI with 100+ deps (mongodb, openid-client, marked, dompurify, jsdom, sharp, undici, etc.) — currently unauditable. This is itself a finding (see Recommendations).

---

## Critical CVEs (CVSS ≥ 9.0 or "critical" severity)

### Rust (Cargo.lock)

| Crate@Version | Advisory | Severity | Reachable? | Fix |
|---|---|---|---|---|
| `rsa@0.9.10` | RUSTSEC-2023-0071 ("Marvin Attack" — timing sidechannel in private-key operations) | High/Critical for crypto | **YES — reachable** in `crates/ruvector-kalshi/src/auth.rs` (PSS signing of API requests). PSS reduces but does not eliminate the timing leak; RSA crate authors mark all 0.9.x as vulnerable. Also pulled by `sqlx-mysql` (likely dormant for our usage). | No fixed release in 0.9.x line; await `rsa 0.10.0`. Mitigations: keep keys local, run on consistent hardware, avoid network-observable signing. |

### npm (`/workspaces/RuVector/npm`)

| Package | Range | Direct? | Notes / Reachability |
|---|---|---|---|
| `@xenova/transformers` (>=2.0.2) → `onnxruntime-web` (<=1.16.0) → `onnx-proto` / `protobufjs` (<=7.5.4) | transitive | No | **Likely dormant** — pulled by `agentdb`/`agentic-flow`; no fix available upstream. Vector is malformed protobuf during model load. |
| `agentdb` (>=1.1.3), `agentic-flow` (<=1.8.15 \|\| >=1.9.1), `aidefence` (*), `lean-agentic` (>=0.2.0), `dspy.ts` (*), `claude-flow` (>=2.0.0-alpha.2) | various | **Yes (direct)** for `aidefence`, `claude-flow`, `dspy.ts` | All chained to the same `@xenova/transformers`/`onnxruntime-web` root cause. `claude-flow` and `dspy.ts` have fixed semver-major versions available. |
| `@google-cloud/redis` (<=3.3.0) → `google-gax` (0.13.5–4.6.1) | transitive | **Yes (direct)** | Fix: upgrade to `@google-cloud/redis@5.2.1` (semver-major). Reachable iff project uses GCP Redis client. |
| `fast-xml-parser` (<=5.6.0) | transitive | No | Fix available; reachable through AWS SDK / cloud SDKs. |
| `handlebars` (4.0.0–4.7.8) | transitive | No | Prototype-pollution / RCE family. Fix available. |

---

## High CVEs (CVSS 7.0–8.9)

### Rust

None of the version-locked Cargo deps map to currently-open High-severity RUSTSEC entries. Notable patches already in place:

- `h2@0.3.27` and `h2@0.4.13` — both ≥ patched against rapid-reset / GOAWAY DoS family (RUSTSEC-2024-0332).
- `openssl@0.10.76` — past RUSTSEC-2025-0022 fix line (0.10.72+).
- `tokio@1.51.0` — past RUSTSEC-2025-0023.
- `curve25519-dalek@4.1.3` — past RUSTSEC-2024-0344.
- `idna@1.1.0` — past RUSTSEC-2024-0421.
- `tar@0.4.45`, `tungstenite@0.24.0`, `rustls@0.23.37`, `ureq@2.12.1` — all past current advisories.

### npm (top-impact direct deps)

| Package | Severity | Range | Fix |
|---|---|---|---|
| `@modelcontextprotocol/sdk` | High | 1.10.0–1.25.3 (cross-client data leak GHSA-345p-7cg4-v4c7) | Upgrade ≥1.26.0 (non-major) — **easy win** |
| `@hono/node-server` | High | <1.19.10 (auth bypass via encoded slashes), <1.19.13 (path traversal) | Upgrade ≥1.19.13 |
| `fastify` | High | <=5.8.2 | Upgrade to `fastify@5.8.5` (semver-major) |
| `node-gyp` | High | <=10.3.1 | Upgrade to `node-gyp@12.3.0` (semver-major) — pulls fixes for `tar`, `cacache`, `make-fetch-happen` |
| `@typescript-eslint/{parser,eslint-plugin}` | High | 6.16.0–7.5.0 (via `minimatch`) | Upgrade to v8.59.1 (semver-major) |
| `@anthropic-ai/claude-code` | High | <=2.1.74 (sandbox escape via symlink, CVSS 10) | Upgrade ≥2.1.75 |
| `undici` | High | 7.0.0–7.23.0 | Patch available |
| `vite` | High | <=6.4.1 \|\| 7.0.0–7.3.1 | Upgrade via `vitest@4.1.5` |

Total: **27 high-severity npm advisories**, the majority transitively fixable by upgrading 5–6 direct deps.

---

## Unmaintained dependencies (RUSTSEC "unmaintained" advisories)

Confirmed present in `Cargo.lock`:

| Crate@Version | Advisory | Reverse deps |
|---|---|---|
| `serde_yaml@0.9.34+deprecated` | RUSTSEC-2024-0320 (unmaintained, archived) | **`rvAgent/rvagent-middleware`** (skill frontmatter parsing) — **directly reachable** |
| `paste@1.0.15` | RUSTSEC-2024-0436 | Heavy use across `gemm-*` (BLAS) and `av-scenechange` (25+ reverse deps) |
| `derivative@2.2.0` | RUSTSEC-2024-0388 | `fusion-blossom@0.2.12` |
| `instant@0.1.13` | RUSTSEC-2024-0384 | `parking_lot@0.11.2`, `parking_lot_core@0.8.6`, bench crates |
| `proc-macro-error@1.0.4` | RUSTSEC-2024-0370 | `tabled_derive`, `validator_derive` |
| `term@0.7.0` | RUSTSEC-2018-0015 | `clap_builder@4.5.60`, `naga`, `prettytable-rs`, `codespan-reporting` |

`atty` is **not** present (good — older RUSTSEC-2021-0145 not applicable).

---

## Outdated pinned versions

| Crate | Current | Recommended | Reason |
|---|---|---|---|
| `reqwest 0.11.27` | coexists with 0.12.28 (12 reverse deps still on 0.11) | unify to `0.12.x` | `reqwest 0.11` is EOL; pulls legacy `hyper 0.14` and `h2 0.3`. `mcp-brain-server` already uses `reqwest = "0.12"` ✓ |
| `hyper 0.14.32` | coexists with 1.9.0 | unify to `1.x` | Legacy branch; only receives critical patches |
| `rustls-pemfile 1.0.4` | coexists with newer | upgrade to `2.x` | Old branch superseded |
| `webpki-roots 0.26.11` | coexists with 1.0.6 | unify to 1.x | Old branch |
| `nix` | 0.26.4 + 0.28.0 + 0.29.0 + 0.31.2 (4 majors) | unify to ≥0.31 | Duplicate-vuln surface |
| `rand` | 0.6.5 + 0.8.5 + 0.9.2 + 0.10.0 | drop 0.6.5 / 0.8.5 if possible | 4 majors in tree |
| `hashbrown` | 0.12 + 0.14 + 0.15 + 0.16 | rationalize | 4 majors |
| `base64` | 0.13.1 + 0.21.7 + 0.22.1 | unify to 0.22 | Old branches |
| `zip` | 1.1.4 + 2.4.2 | drop `zip 1.1.4` (RUSTSEC family) | Two majors |
| `mcp-gate/Cargo.toml`: `base64 = "0.21"`, `thiserror = "1.0"` | older pins | bump to `base64 = "0.22"`, `thiserror = "2.0"` (mcp-brain-server already uses 2.0) | Consistency |
| `libsqlite3-sys 0.30.1` | bundles SQLite ~3.46 | upgrade `rusqlite`/`sqlx-sqlite` so bundled SQLite ≥3.49.2 | CVE-2025-29087 (heap overflow in `concat_ws`) — likely dormant unless that function is used with attacker input |

### Cargo.toml spot-check

- `crates/mcp-brain-server/Cargo.toml`: clean. Modern pins (`reqwest 0.12`, `axum 0.7`, `tokio 1.41`, `thiserror 2.0`, `ed25519-dalek 2`, `chrono 0.4`).
- `crates/mcp-gate/Cargo.toml`: `tokio = "1.35"`, `thiserror = "1.0"`, `base64 = "0.21"` — bump suggested but no security delta.
- `crates/ruvector-router-ffi/Cargo.toml`: minimal NAPI shim, deps are workspace-pinned. Clean.

---

## License risks

No GPL/AGPL crates were detected by name in `Cargo.lock`. Project license is MIT (root `package.json`) and MIT/Apache-2.0 (Cargo crates). No quick-grep hits for `gpl`, `agpl`, `gnutls`, `readline`, `libssh2-sys-gpl`. **No action required**, but a full `cargo deny check licenses` should be run when `cargo` is available to be exhaustive.

---

## Recommended upgrade plan (priority order)

1. **(P0) `ui/ruvocal/`: commit a `package-lock.json`.** Currently zero visibility into the SvelteKit UI's transitive vulns. Likely large attack surface (mongodb, openid-client, jsdom, marked, dompurify).
2. **(P0) Remove or replace `rsa@0.9.10` private-key signing in `ruvector-kalshi`.** Marvin Attack is a real timing sidechannel. Options: (a) use `ring` for RSA-PSS, (b) wait for `rsa 0.10`, (c) document that signing happens on isolated hardware.
3. **(P0) npm direct deps:** upgrade `@google-cloud/redis` → 5.2.1, `@modelcontextprotocol/sdk` → ≥1.26.0, `claude-flow` → fixed line, `dspy.ts` → 0.1.3, drop `aidefence`/`lean-agentic` if not actively used (no fix available).
4. **(P1) Replace `serde_yaml` in `rvagent-middleware/skills.rs`** with `serde_yml` (community fork) or `serde_yaml_ng`. RUSTSEC-2024-0320, directly reachable.
5. **(P1) Unify `reqwest` to 0.12 across the workspace** (drops legacy `hyper 0.14`, `h2 0.3`, `rustls-pemfile 1.0`, `webpki-roots 0.26`). 12 crates need touching — see reverse-deps list above.
6. **(P1) npm `@hono/node-server` → ≥1.19.13**, `fastify` → 5.8.5, `node-gyp` → 12.3.0, `@anthropic-ai/claude-code` → ≥2.1.75, `@typescript-eslint/*` → 8.x.
7. **(P2) Drop legacy duplicates:** `zip 1.1.4`, `nix 0.26/0.28`, `rand 0.6.5/0.8.5`, `base64 0.13/0.21`, `hashbrown 0.12/0.14`. Reduces duplicate-vuln surface.
8. **(P2) Upgrade `libsqlite3-sys`** so bundled SQLite ≥ 3.49.2 (CVE-2025-29087).
9. **(P3) Replace unmaintained transitive crates** (`paste`, `derivative`, `instant`, `proc-macro-error`, `term`) where direct upstreams have moved on. Most are deep transitives in BLAS/wgpu-naga/clap; track upstream rather than fork.
10. **(P3) Install `cargo-audit` and add CI job** running `cargo audit --deny warnings` against `Cargo.lock` on every PR. Add `npm audit --audit-level=high --omit=dev` likewise.

---

## Files referenced

- `/workspaces/RuVector/Cargo.lock`
- `/workspaces/RuVector/npm/package-lock.json`
- `/workspaces/RuVector/package.json`
- `/workspaces/RuVector/npm/package.json`
- `/workspaces/RuVector/ui/ruvocal/package.json` (no lockfile)
- `/workspaces/RuVector/crates/mcp-brain-server/Cargo.toml`
- `/workspaces/RuVector/crates/mcp-gate/Cargo.toml`
- `/workspaces/RuVector/crates/ruvector-router-ffi/Cargo.toml`
- `/workspaces/RuVector/crates/ruvector-kalshi/src/auth.rs` (rsa reachability)
- `/workspaces/RuVector/crates/rvAgent/rvagent-middleware/src/skills.rs` (serde_yaml reachability)
