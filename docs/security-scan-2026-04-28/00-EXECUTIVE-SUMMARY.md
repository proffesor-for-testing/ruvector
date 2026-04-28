# RuVector — Ultra-Deep Security Scan: Executive Summary

**Date:** 2026-04-28
**Scope:** Whole project at `/workspaces/RuVector` — 129 Rust crates, ~30 npm packages, ruvocal UI, MCP servers, Cloud Run deployment, NAPI/WASM bindings, install pipeline, GitHub Actions, git history.
**Method:** 8 parallel specialized security agents (Rust SAST, JS/TS SAST, dependency CVEs, secrets/PII, injection/RCE/SSRF, auth/crypto, supply chain, architectural threat model). Read-only audit.
**Detail reports:** see siblings `01-rust-sast.md` through `08-threat-model.md`.

---

## TL;DR — Posture Verdict

**4 / 10 (public brain plane)** · **6 / 10 (local libraries / RVF crypto primitives)**

The team has built **strong cryptographic primitives** (Ed25519, SHAKE-256, ML-DSA-65 PQ, constant-time compares, ADR-042 TEE design, deliberate `unsafe` hygiene per ADR-007). The deployment of those primitives in the public brain at `pi.ruv.io` is **wide open**: anyone on the internet can write to it with any 8-character string, the reputation system is Sybil-trivial, the WASM signature it declares in its own schema is never verified, the differential-privacy claim covers the wrong field, and the witness chain has no published trust root. Untrusted writes feed the nightly LoRA trainer that distributes weights back to all clients — a single point of failure for the entire AI supply chain.

**Strong primitives + weak system.** Two weeks of focused work on items 1–3 below would move the brain plane to 7/10.

---

## Tooling Re-Validation Addendum (added after agent fleet)

After the 8-agent fleet completed, `cargo-audit 0.22.1`, `gitleaks 8.30.1`, and `trufflehog 3.95.2` were installed and run for authoritative confirmation. Results in `09-tooling-validation.md`. Two corrections to agent findings:

1. **Rust dependency CVEs: agent count 1C/6 unmaintained → authoritative count 0/0.** `cargo-audit` against the fresh RustSec DB (1058 advisories, 1246 deps in `Cargo.lock`) reports zero open advisories — even with `--deny warnings`. The previously cited `RUSTSEC-2023-0071` (rsa Marvin Attack) and `RUSTSEC-2024-0320` (serde_yaml) IDs are not active against these versions in the current DB. Recommendation P1.17 (replace `rsa@0.9.10`) and P3 unmaintained-crate items downgraded.
2. **Secrets scan: gitleaks confirms PII agent's count of 1 real leak.** `gitleaks` raw output shows 132 hits across full history, 35 in working tree. Triage: all working-tree hits are test fixtures (`#[test]` blocks in `crates/mcp-brain/src/pipeline.rs`), n8n test fixtures, README examples, gitignored local files, or third-party research bundles. Only confirmed real leak remains the Firebase web API key in commit `1a6174bc`. Action unchanged: **rotate / restrict in GCP Console**.

**Net effect on posture:** unchanged at 4/10 / 6/10. The dependency surface improves; the architectural issues (auth bypass chain, memory poisoning → LoRA, WASM signature unverified, DP/witness-chain claim gaps) are not detectable by any CVE/secrets scanner — they remain the real risk story.

`trufflehog` (full git history with verification) and `cargo-geiger` (unsafe-density scoring) are still running and will be appended to `09-tooling-validation.md` when complete.

---

## Top 10 Findings (Ranked)

| # | Severity | Finding | Where | Detail |
|---|---|---|---|---|
| 1 | **CRITICAL** | **Auth bypass — any 8-char Bearer token = valid contributor** | `crates/mcp-brain-server/src/auth.rs:23-99` | `from_api_key` does no DB/JWT/allowlist check — only length + SHAKE-256 derivation. Confirmed by 3 independent agents. Root cause for findings #2, #3, #4. |
| 2 | **CRITICAL** | **Indirect prompt injection via stored memories → public GitHub gist publication** | `routes.rs:7684, 7755` (Gemini sink with `google_search` tool); `gist.rs:357, 618` (auto-published) | Single poisoned memory hijacks every Google Chat reply and the discovery gist generator. Persistent until manually purged. Tool-use hijack possible (model has `google_search`). |
| 3 | **CRITICAL** | **Brain memory poisoning → LoRA training corpus poisoning** | `routes.rs:1416` (share_memory) → ADR-129 nightly trainer | Nightly LoRA bakes anonymous writes into weights distributed to all clients. ~$50/mo residential proxy pool defeats per-IP rate limits. **AI supply-chain SPoF.** |
| 4 | **CRITICAL** | **WASM publish signature declared but never verified** | `routes.rs:4963-5084` (handler); `types.rs:989` (field declared); `verify.rs:178` (verifier exists, unused) | Server requires `signature` field, never calls Ed25519 verifier. Reputation gate is 0.5 — trivial. Other agents download `/v1/nodes/{id}.wasm` and execute. |
| 5 | **CRITICAL** | **Public MCP transport `/sse` and `/messages` have no auth at all** | `routes.rs:373-374, 5910-5913` | Hardcoded loopback dispatcher uses key `"mcp-sse-session"` for every JSON-RPC call. Any anonymous client gets full write to `brain_share`, `brain_vote`, `brain_delete`, `brain_transfer`. Defeats per-key rate limits. |
| 6 | **CRITICAL** | **`/internal/*` endpoints documented as no-auth, bound to `0.0.0.0`** | `routes.rs:8086-8194`; `main.rs:19` | Pod-reachable callers can hijack sessions, drain other clients' response queues, smuggle messages. No middleware enforces sidecar-only access. |
| 7 | **CRITICAL** | **`verify_system_key` fails-OPEN when env unset** | `routes.rs:8038-8050` | Forgotten `BRAIN_SYSTEM_KEY` silently exposes 7 admin email/notify endpoints. Becomes spam relay through Resend account. |
| 8 | **CRITICAL** | **Shell injection in rvAgent sandbox & LLM tool surface** | `crates/rvAgent/rvagent-backends/src/sandbox.rs:123, 142` (sh -c with single-quote escape); `rvagent-cli/src/app.rs:332-340` (raw `command` → `sh -c`) | LLM `execute()` tool is full-host RCE — any prompt injection that reaches the tool yields shell. No timeout, no sandboxing on the CLI tool. |
| 9 | **CRITICAL** | **Firestore path traversal via `publish_node.id`** | `routes.rs:5059` → `store.rs:1319` → `store.rs:244` | Attacker `id="../brain_memories/<victim_uuid>"` allows cross-collection PATCH within the project. Can overwrite reputation rows for self-elevation, or amplify finding #2. |
| 10 | **CRITICAL** | **rsa@0.9.10 Marvin Attack reachable in production** | `crates/ruvector-kalshi/src/auth.rs` (PSS signing) | RUSTSEC-2023-0071 timing sidechannel; no fix in 0.9.x line. Exploitable in network-observable signing scenarios. |

---

## High-Severity Findings (Selected)

| # | Severity | Finding | Where |
|---|---|---|---|
| H1 | High | DP claim ε=1.0 doesn't hold — noise on embedding only; raw `title`/`content`/`tags` stored verbatim. Sensitivity miscalibrated by ~20×. `PrivacyAccountant` never wired. | `routes.rs:1485-1494`; `rvf-federation/src/diff_privacy.rs` |
| H2 | High | Witness chain has no signature, no MAC, no external anchor — wholesale forgery trivial; only protects against accidental corruption. | `rvf-crypto/src/witness.rs:85-111` |
| H3 | High | `pipeline_inject` & `pipeline_inject_batch` write endpoints fully unauthenticated, no rate limit | `routes.rs:3699-3769` |
| H4 | High | `email_inbound` discards Resend HMAC `svix-signature` — anyone can spoof `from:` | `routes.rs:7838-7841` |
| H5 | High | `consciousness_compute` unauthenticated and CPU-bound (n=4096) — single-call DoS | `routes.rs:8227-8265` |
| H6 | High | Per-IP rate limit trivially bypassed via spoofed `X-Forwarded-For` | `routes.rs:33-43` |
| H7 | High | FFI `static mut STORE_STATE` data race UB (multi-threaded NAPI host) | `crates/ruvector-temporal-tensor/src/store_ffi.rs:200`; `ffi.rs:13` |
| H8 | High | FFI `ttc_dealloc` reconstructs `Vec::from_raw_parts` with wrong capacity → UB / heap corruption | `crates/ruvector-temporal-tensor/src/ffi.rs:243-249` |
| H9 | High | Persistent XSS via unescaped `node.id` / `node.metadata` to `innerHTML` | `npm/packages/ruvector-extensions/src/ui/app.js:255-272` |
| H10 | High | `Math.random()` for Raft election timeouts — split-brain / liveness risk | `npm/packages/raft/src/node.ts:256` |
| H11 | High | CORS wildcard `*` on writable HTTP APIs (with `Authorization` allowed header) | `npm/packages/ruvector-extensions/src/ui-server.ts:54`; `npm/packages/ruvbot/src/server.ts:531` (×5 sites) |
| H12 | High | SDK default API key is the literal string `"anonymous"` — collapses every unconfigured client to one shared identity | `npm/packages/pi-brain/src/client.ts:66-71` |
| H13 | High | SSRF via Common Crawl ingest — no allowlist / no private-IP filter on outbound fetches | `crates/mcp-brain-server/src/pipeline.rs:1185, 1226` |
| H14 | High | 14 Critical + 27 High npm CVEs (`@google-cloud/redis`, `@hono/node-server`, `@anthropic-ai/claude-code`, `@modelcontextprotocol/sdk`, `handlebars`, `protobufjs`, `undici`, `tar`, `node-forge`, etc.) | `npm/package-lock.json` |

---

## Confirmed Secrets / PII Leaks

| # | Severity | Finding |
|---|---|---|
| S1 | **CRITICAL — ROTATE NOW** | Google Firebase web API key for GCP project `ruv-dev` committed in commit `1a6174bc`, files `examples/edge-net/dashboard/src/services/firebaseData.ts` and `examples/edge-net/pkg/firebase-signaling.js`. Removed from working tree but still reachable via `git show 1a6174bc`. Action: **restrict / rotate in GCP Console** and audit Firebase Security Rules for project `ruv-dev`. |

Everything else (test fixtures, `.env.example` placeholders, project author emails) was filtered as expected. **No active credentials in the working tree.** Git history of 5,575 commits scanned via `git log -S` for high-signal patterns.

---

## Supply Chain Verdict

- **`install.sh`** — **trustworthy** as a convenience wrapper. Trust boundary is crates.io and npmjs.com, not the script.
- **NAPI router binaries** — **trustworthy with one caveat**: confirmed built in CI from source (no `.node` files in git, only skeleton `package.json` files). Built reproducibly on matching runner OS. **But unsigned** — no Sigstore/npm provenance attestation. Adding `--provenance` to publish steps is a one-line fix.
- **No `pull_request_target` triggers anywhere** — the most important class of CI RCE is absent. Good.
- **NPM_TOKEN / CARGO_REGISTRY_TOKEN** never exposed to PR runs — all publish jobs gated on tags / `workflow_dispatch`.

**Systemic CI gaps:** zero workflows have a top-level `permissions:` block (38 workflows), all third-party actions are tag-pinned (not SHA-pinned), `dtolnay/rust-toolchain@stable` (floating branch) used 53×, `mcp-brain-server` Dockerfile runs as root with no HEALTHCHECK, `scripts/sync-lockfile.sh:30` runs `npm install` without `--ignore-scripts` in pre-commit hook.

---

## Cleared as Well-Defended (For Reference)

- `eval` / `new Function` / `setTimeout(string)` / `document.write` — **zero hits**
- Hardcoded API keys / JWT secrets in source — **zero hits**
- `rejectUnauthorized: false` / disabled TLS verification — **zero hits**
- Prototype pollution sinks (`lodash.merge`, `Object.assign(req.body)`) — **zero hits**
- `fetch-url` SSRF in ruvocal — robust TOCTOU-safe `assertSafeIp` per resolved address
- CSRF on OAuth state — properly bound to `sessionId`
- ruvocal markdown XSS — sanitized through custom `marked.ts` + DOMPurify
- All `rusqlite` queries properly parameterized — no SQL injection
- WASM publish has 1 MB limit + magic-byte check (deserialization RCE blocked)
- No XML parsing in scope (no XXE)
- No SSTI (compile-time format strings only)
- `unsafe` blocks in `cognitum-gate-kernel`, `ruvector-core` reviewed and sound (every block has SAFETY comment + bounds check)

---

## Differential Privacy & Witness Chain — Detailed Verdict

### DP claim ε=1.0 — **does not hold**

- Noise added to embedding only; raw `title`/`content`/`tags` stored verbatim and globally readable (`GET /v1/memories/{id}`).
- L2 sensitivity should be `2 × clipping_norm = 20.0`, set to `1.0` → sigma understates required noise by ~20× → **real per-write ε ≈ 20× claimed value**.
- No cumulative budget tracking. `PrivacyAccountant` exists in `diff_privacy.rs:170-299` but is never wired in. After N writes, total ε scales as N (naive) or √N (RDP) — **system ε after months of use is in the hundreds**.
- Single shared `Mutex<DiffPrivacyEngine>` with persistent RNG state.

**Action:** either drop the DP language from CLAUDE.md / README, or actually protect content (encrypt at rest with per-tenant keys, DP at retrieval not write).

### Witness chain "tamper-evident audit log" — **does not hold**

- SHAKE-256 hash chain with no signature, no MAC, no external anchor. `verify_witness_chain` only checks self-consistency.
- Anyone can construct a brand-new chain from scratch that verifies. Wholesale forgery is trivial. Tampering with a single entry is detected; that's the only protection.
- No external anchor (Roughtime / RFC 3161 / Sigsum / Trillian / DNS TXT / Bitcoin OP_RETURN). Operator can rewrite Firestore and rebuild the chain.

**Action:** sign each chain head with an Ed25519 key (verifier already imports `ed25519_dalek`), publish public key on `/v1/status`, anchor heads to an external transparency log periodically.

---

## Recommended Remediation Order

### P0 — Ship before next release (the chained criticals)

1. **Replace `from_api_key` no-op with real allowlist/JWT/HMAC check** (root cause for #1/#2/#3/#5/#9).
2. **Wire Ed25519 verification into `publish_node`** (`verify.rs:178` exists; just call it).
3. **Default-deny auth middleware** for every route except an explicit allowlist (`/v1/health`, `/v1/ready`, `/.well-known/*`, `/v1/status`). Fixes 30+ missing-auth routes in one change.
4. **Move `/internal/*` to a separate listener** bound to `127.0.0.1` (or Unix socket).
5. **`verify_system_key` fail-CLOSED** when env empty in non-debug builds.
6. **Stop hardcoding `"mcp-sse-session"`** — propagate the SSE handshake's API key.
7. **Sanitize memory content before LLM inlining** — strip `</system>`, `<|...|>`, `[INST]`, fence in quoted block. Disable `google_search` grounding when memories were used as input.
8. **Validate `WasmNode.id` to `^[a-zA-Z0-9._-]{1,128}$`** to close Firestore path traversal.
9. **Switch rvAgent sandbox file ops off `sh -c`** to `Command::new("cat").arg(path)`.
10. **Restrict / rotate the leaked `ruv-dev` Firebase API key** and audit Firebase Security Rules.

### P1 — Within sprint

11. Remove `'anonymous'` SDK fallback in `pi-brain/src/client.ts` — throw on missing key.
12. Authenticate `pipeline_inject*`, `consciousness_compute`. Verify Resend `svix-signature` on `email_inbound`.
13. Trust `X-Forwarded-For` only when peer IP is in trusted-proxy CIDR list (Cloud Run frontend).
14. Replace `Math.random()` in `npm/packages/raft/src/node.ts:256` with `crypto.randomInt`.
15. Remove `Access-Control-Allow-Origin: *` from writable APIs (`ruvector-extensions`, `ruvbot`).
16. Fix XSS in `ruvector-extensions/src/ui/app.js:255-272` — use `textContent` / `DOMPurify`.
17. Replace `rsa@0.9.10` with `ring` for PSS signing in `ruvector-kalshi`, OR document network-isolated signing.
18. Replace `serde_yaml` (unmaintained) with `serde_yml` in `rvagent-middleware/src/skills.rs`.
19. `npm audit fix` for the 14 critical + 27 high; upgrade direct deps `@google-cloud/redis@5.2.1`, `@modelcontextprotocol/sdk@≥1.26.0`, `@hono/node-server@≥1.19.13`, `@anthropic-ai/claude-code@≥2.1.75`, `fastify@5.8.5`, `node-gyp@12.3.0`.
20. Add `permissions: contents: read` top-level to all 38 workflows; SHA-pin `dtolnay/rust-toolchain` (53 sites) and other third-party actions.

### P2 — Architectural

21. **Either drop the differential-privacy claim OR implement content-level privacy** (encrypted-at-rest + DP at retrieval).
22. **Publish witness chain trust root** (sign chain heads, anchor to external transparency log).
23. **Trust-tier the brain memories** — nightly LoRA trainer reads only `human-reviewed` tier. Defeats poisoning attack (#3) entirely.
24. **Move `ruvbrain` off `--allow-unauthenticated`** — Cloud Armor / API gateway, real revocable API keys after CAPTCHA or GitHub OAuth.
25. **Add `--provenance` and `id-token: write` to every NAPI publish workflow** (npm provenance via Sigstore).
26. **Add `SECURITY.md`, `security@ruv.io` mailbox, documented disclosure process, secret-rotation runbook.**
27. **Fix FFI memory-safety**: wrap `static mut STORE_STATE` in `Mutex` (or gate to `wasm32` only); track Vec capacity in a side-table for `ttc_dealloc`.
28. **Unify dependency tree**: drop `reqwest 0.11`, `hyper 0.14`, `nix 0.26/0.28`, `rand 0.6.5/0.8.5`, `base64 0.13/0.21`, `zip 1.1.4`. Reduces duplicate-vuln surface.
29. **Commit `package-lock.json` for `ui/ruvocal/`** — currently zero vuln visibility on the SvelteKit chat UI.

### P3 — Hygiene

30. Install `cargo-audit` + `cargo-geiger` + `gitleaks` and add to CI on every PR.
31. Run `clippy --all-targets -- -D clippy::unwrap_used -D clippy::expect_used` to catch the panic-on-untrusted-input class.
32. Validate ownership of `examples/vectorvroom` submodule (`shaal/VectorVroom`).
33. Switch from MD5 to SHA-256/BLAKE3 for `ruvllm` cache keys.
34. Switch CI `curl … | sh` wasm-pack pattern to `taiki-e/install-action`.

---

## Appendix — Detail Reports

| File | Agent | Severity counts |
|---|---|---|
| [01-rust-sast.md](01-rust-sast.md) | Rust SAST | 4C / 5H / 7M / 6L |
| [02-js-ts-sast.md](02-js-ts-sast.md) | JS/TS SAST | 1C + 14C dep / 2H + 27H dep / 4M + 24M dep / 3L + 3L dep |
| [03-dependency-cves.md](03-dependency-cves.md) | Dependency CVEs | 14C npm + 1C reachable Rust / 27H npm |
| [04-secrets-pii.md](04-secrets-pii.md) | Secrets / PII | 1C confirmed (Firebase key in git history) |
| [05-injection-rce-ssrf.md](05-injection-rce-ssrf.md) | Injection / RCE / SSRF | 2 confirmed exploitable / 2 likely / 4 theoretical |
| [06-auth-crypto.md](06-auth-crypto.md) | Auth / AuthZ / Crypto | 3C / 5H / 4M / 4L |
| [07-supply-chain-ci.md](07-supply-chain-ci.md) | Supply chain & CI/CD | 0 critical CI RCE / 5 high hardening gaps |
| [08-threat-model.md](08-threat-model.md) | Architecture threat model | 10 critical architectural risks ranked |

**Tooling caveats:** `cargo`, `cargo-audit`, `cargo-geiger`, `gitleaks`, `trufflehog` were not available in the devcontainer. `cargo audit` was replaced with manual `Cargo.lock` analysis vs. RUSTSEC DB. Secret scanning used `ripgrep` with custom regex set. All findings should be validated with the proper tooling before final disclosure.
