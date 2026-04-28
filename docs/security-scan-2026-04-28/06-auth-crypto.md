# Auth / AuthZ / Crypto Review

Scope: `crates/mcp-brain-server/`, `crates/mcp-gate/`, `crates/cognitum-gate-kernel/`, `crates/rvf/rvf-crypto/`, `crates/rvf/rvf-federation/`, `npm/packages/pi-brain/`, `npm/packages/cloud-run/`. Read-only review.

## Summary

The `pi.ruv.io` brain server (`crates/mcp-brain-server`) has a partially-applied authentication model. A correct, constant-time API-key extractor (`AuthenticatedContributor`, `auth.rs`) is implemented but is selectively applied per-route — only 42 of ~80 handlers use it. Several high-impact and write endpoints (full MCP transport, pipeline injection, internal queue/session ops, consciousness compute, inbound email) are completely unauthenticated. The "system key" gate fails open when `BRAIN_SYSTEM_KEY` is unset, the SDK defaults its key to the literal string `"anonymous"`, and the loopback MCP dispatcher hard-codes the key `"mcp-sse-session"` for every JSON-RPC tool call — three independent bypasses that converge on the same conclusion: in practice, write authentication is not effectively enforced.

Cryptography is mostly modern (SHAKE-256, ed25519, subtle::ConstantTimeEq, base64 STANDARD), no MD5/SHA1/DES/RC4/ECB found. The differential-privacy and witness-chain claims advertised in `CLAUDE.md` (ε=1.0; tamper-evident audit log) **do not hold up to inspection** — see the dedicated section.

No JWTs are in use anywhere in the brain server, so the entire JWT class of bugs (alg=none, RS/HS confusion, etc.) is not applicable. No TLS verification is disabled in any reviewed code.

## Authentication findings (with severity)

### CRITICAL — Public MCP transport (`/sse`, `/messages`) requires no auth and proxies as a privileged shared identity
`crates/mcp-brain-server/src/routes.rs:373-374` registers `/sse` and `/messages` with no `AuthenticatedContributor` extractor. Inside the handler at `routes.rs:5903-5913`, every MCP `tools/call` is forwarded to the loopback REST API with a hard-coded key:
```rust
let api_key = args.get("_api_key").and_then(|k| k.as_str())
    .unwrap_or("mcp-sse-session");
```
Result: any unauthenticated client that opens an SSE session can call every MCP tool — `brain_share`, `brain_vote`, `brain_delete`, `brain_transfer`, `brain_consciousness_compute` — and the downstream REST handler treats them as the single shared pseudonym derived from `"mcp-sse-session"`, bypassing per-key rate limits and contributor accountability. Self-voting at scale also lets this shared identity climb the reputation gate (≥0.5) used to authorize WASM `publish_node`.

### CRITICAL — `/internal/*` endpoints documented as auth-free are exposed on the public listener
`routes.rs:8086-8194` and the comment at `8062-8064` ("These are service-to-service endpoints used by the SSE proxy ... No authentication required"). The handlers `internal_queue_push`, `internal_queue_drain`, `internal_session_create`, `internal_session_delete` are all registered on the same `Router` (`routes.rs:406-409`) that `main.rs:19` binds to `0.0.0.0:PORT`. There is no middleware or path-prefix filter restricting them to a sidecar. Any caller who can reach the pod can:
- Push arbitrary messages into any session's mpsc channel (`POST /internal/queue/push`, line 8091) — this is a server-side request smuggling primitive.
- Drain any session's response queue (`GET /internal/queue/drain`) — leaks responses intended for other clients.
- Create or delete any session (`POST /internal/session/create`, `DELETE /internal/session/:id`) — DoS and session hijack.

### CRITICAL — `verify_system_key` is fail-open (`routes.rs:8038-8057`)
```rust
let system_key = std::env::var("BRAIN_SYSTEM_KEY").unwrap_or_default();
if system_key.is_empty() {
    return Ok(()); // dev mode bypass
}
```
All seven endpoints gated by `verify_system_key` (`notify_test`, `notify_status`, `notify_send`, `notify_welcome`, `notify_help`, `notify_digest`, `notify_opens`) become fully open if the env var is missing in production. There is no startup check to abort if the key is absent in a non-dev environment. Impact: ability to spam emails through the project's Resend account at the operator's expense, and to enumerate notification recipients.

### HIGH — SDK default API key is the constant string `"anonymous"`
`npm/packages/pi-brain/src/client.ts:66-71`:
```ts
this.apiKey = options?.apiKey ?? process.env.PI ?? process.env.BRAIN_API_KEY ?? 'anonymous';
```
`"anonymous"` is 9 chars, so it satisfies `MIN_API_KEY_LEN = 8` in `auth.rs:53`. Every SDK user without a configured key collapses to a single SHAKE-256-derived pseudonym, sharing one rate-limit bucket and one reputation. Anti-Sybil and per-contributor reputation are meaningless for that population.

### HIGH — `pipeline_pubsub_push` trusts ambient Cloud Run IAM but performs no in-process validation
`routes.rs:3772-3823` carries the comment "No Bearer auth required (Cloud Run validates Pub/Sub OIDC tokens automatically)" but the code performs no OIDC validation. This is correct only when Cloud Run is configured with `--no-allow-unauthenticated` and the `roles/run.invoker` binding is restricted to the Pub/Sub push subscription's service account. There is no defense-in-depth (no in-process JWT verification, no shared secret), and no comment in the deploy scripts pinning this contract — a single misconfigured deployment exposes a high-rate write path.

### HIGH — `pipeline_inject` and `pipeline_inject_batch` write endpoints have no authentication
`routes.rs:3699-3769` register-and-handle as `pub` `POST /v1/pipeline/inject` and `/inject/batch` with only `State` and `Json` extractors (`routes.rs:361-362`). Anyone reaching these endpoints can bulk-inject memories into the brain at up to 100/req. No rate limit, no contributor binding, and the inserted content bypasses the per-key write-rate counter that exists on `share_memory`.

### HIGH — `email_inbound` does not verify the Resend webhook signature
`routes.rs:7838-7841`:
```rust
async fn email_inbound(State(state): State<AppState>, _headers: HeaderMap,
                      Json(payload): Json<ResendInboundPayload>) -> ...
```
The `_headers` argument is ignored. Resend supplies an HMAC-signed `svix-signature` header for inbound webhooks; without verifying it, anyone can POST a forged "from" address and trigger the brain to execute commands (`search`, `subscribe`, etc.) and reply to that spoofed address — useful for spam relay and command injection into the brain.

### HIGH — `consciousness_compute` is unauthenticated and CPU-bound
`routes.rs:8227-8265` accepts up to `n=4096` (`max_elements=12`) and runs CES / IIT-4 algorithms. With no auth and no rate limit, a single anonymous caller can saturate CPU.

### MEDIUM — `extract_client_ip` blindly trusts `X-Forwarded-For`
`routes.rs:34-41` reads the first comma-separated value from any client-supplied `X-Forwarded-For` header. This header is supposed to be set by the Cloud Run frontend, but Axum performs no check that the request actually arrived through the GFE. A direct caller can spoof the header to bypass the per-IP write rate limit (`share_memory`, `routes.rs:1435-1441`) and the per-IP anti-Sybil vote dedup (`vote_memory`, `routes.rs:2278-2289`). Use the connecting socket's `ConnectInfo` for the trust boundary, then accept `XFF` only when the immediate peer is the GFE.

### LOW — `MIN_API_KEY_LEN = 8` is too short
`auth.rs:53`. Eight printable bytes is ~48 bits of entropy at best (much less if user-chosen). For a public collective brain, raise to 32 random bytes (256 bits), and reject keys that fail an entropy check.

### LOW — `_reserved` system pseudonym leak via `system_seed`
`auth.rs:43-49` constructs a `system` contributor with hard-coded pseudonym `"ruvector-seed"`. If any code path returns `is_system=true` from a request not gated by `BRAIN_SYSTEM_KEY`, a caller becomes the privileged contributor. Currently only `from_request_parts` sets `is_system`, but `system_seed()` is `pub` — ensure it stays internal.

## Authorization findings

### HIGH — No ownership check on `share_memory` votes, only a per-IP dedup
`routes.rs:2250-2358`. The author of a memory is allowed to self-vote (`is_author` skips IP dedup, lines 2277-2289), and reputation increases on every up-vote (line 2304-2308). Combined with the `mcp-sse-session` shared-identity bypass and the `anonymous` SDK default, a single attacker can mint reputation cheaply, enabling them to publish WASM nodes (`publish_node` reputation gate ≥ 0.5, line 4984).

### HIGH — IDOR on memory deletion is mitigated by store-layer check, but only if `pseudonym` is correctly authenticated
`routes.rs:2361-2391` calls `state.store.delete_memory(&id, &contributor.pseudonym)`. The store enforces ownership by pseudonym. Combined with the auth bypasses above, anyone using the shared `mcp-sse-session` or `anonymous` identity can delete memories created by anyone else who shares that identity — including all SDK callers without a configured key.

### MEDIUM — No tenant isolation in any reviewed code
There is no concept of an organization or tenant. All memories are in one global pool, scoped only by per-contributor pseudonym. This is by design for the "shared brain" but should be documented as a property — there is no protection against one user's memory being returned in another user's search.

### MEDIUM — `revoke_node` and `delete_memory` perform `pseudonym ==` ownership checks
`routes.rs:5088-5097`, `2361-2391`. The store-side comparison is via plain string equality; not timing-sensitive (pseudonym is not a secret) — acceptable. Noted only for completeness.

## Cryptographic weaknesses

Mostly clean. Specific issues:

### MEDIUM — `DiffPrivacyEngine` RNG is `StdRng` seeded from `thread_rng`, with a public `with_seed(u64)` shortcut
`crates/rvf/rvf-federation/src/diff_privacy.rs:51`:
```rust
rng: StdRng::from_rng(rand::thread_rng()).unwrap(),
```
`thread_rng` is itself OsRng-seeded so the production path is fine, but the public method `with_seed(mut self, seed: u64) -> Self` (line 75) replaces the RNG with a deterministic 64-bit seeded ChaCha. This is intended for tests, but there is no `#[cfg(test)]` gate and no naming hint that it kills the privacy guarantee. A future caller misusing it (or a test fixture leaking into prod) silently turns DP into a sham.

### LOW — `Math.random()` for connection IDs in `cloud-run/vector-client.ts:124` and `streaming-service-optimized.{ts,js}`
Uses non-CSPRNG for `conn-${ts}-${rand}` IDs. These are not security tokens (not used for authn or authz, only for log correlation in a connection pool), so impact is limited to potential ID collisions under load. Switch to `crypto.randomUUID()` for hygiene.

### LOW — `verify.rs:375` uses `rand::thread_rng` for ed25519 signing key generation in tests only
This is in `#[cfg(test)]` (line 372 onward) and would not appear in production paths. No issue.

### INFORMATIONAL — No TLS termination misconfig found
No use of `rejectUnauthorized: false`, `danger_accept_invalid_certs`, or insecure TLS settings in any reviewed code path.

## Differential privacy / witness chain review (verified vs. claimed)

### Differential privacy: claim ε=1.0, reality much weaker

`CLAUDE.md` states "Brain has differential privacy (ε=1.0)". `crates/rvf/rvf-federation/src/diff_privacy.rs` and `routes.rs:1485-1494` (`add_noise` on the embedding before storage) are the implementation. Findings:

1. **The DP engine adds noise only to the embedding vector, not to the title, content, or tags, which are stored verbatim.** A DP claim over an embedding is meaningless when the original text is also persisted; `share_memory` writes both. This is the dominant flaw — the privacy guarantee is rhetorical, not mathematical.

2. **Privacy budget is per-call, not cumulative.** `DiffPrivacyEngine` is constructed once at startup with ε=1.0 (`routes.rs:193`) and `add_noise` is called per write. There is no consultation with `PrivacyAccountant.can_afford()`. Total ε after N writes scales roughly as N under naive composition (and as O(√N) under RDP), so after even a few hundred writes the effective ε is in the hundreds. The advertised "ε=1.0" is the per-record bound, not the system bound.

3. **Sensitivity is set to 1.0 with `clipping_norm=10.0`.** L2 sensitivity of two adjacent embeddings under L2 clipping at norm c is 2c, i.e. 20.0, not 1.0. The Gaussian sigma formula `σ = sensitivity·√(2·ln(1.25/δ))/ε` therefore under-estimates the required noise by a factor of ~20. The actual ε achieved is ~20× larger than reported.

4. **Gaussian mechanism uses the classical Dwork bound, valid only for ε ≤ 1.** With ε exactly 1.0 the bound is borderline; the tighter analytical Gaussian (Balle–Wang, 2018) is not used.

5. **`PrivacyAccountant` is implemented (`diff_privacy.rs:170-299`) but never wired in.** No code calls `record_gaussian` or `is_exhausted` — it exists only in benches and tests.

**Verdict: the ε=1.0 DP claim does not hold.** Real per-write ε is ~20× the configured value, the budget is not tracked, and the original plaintext is stored alongside the noised embedding which makes the DP guarantee meaningless for the field that actually matters.

### Witness chain: claim "tamper-evident cryptographic audit trail", reality unsigned hash chain

`crates/rvf/rvf-crypto/src/witness.rs` defines a chain where `entry[i].prev_hash = SHAKE-256(encode(entry[i-1]))`. There is **no signature, no MAC, no external anchor**. Findings:

1. **No keying material is involved.** `verify_witness_chain` (line 85-111) only checks self-consistency of the hash chain. Anyone (including the brain server itself, or any party that obtains a chain blob) can construct a brand-new chain with arbitrary `action_hash` values and `timestamp_ns` values that verifies just as well. Tampering with one entry is detected; wholesale forgery is not.

2. **No anchoring to an external clock or transparency log.** Timestamps are server-supplied (`timestamp_ns: now_ns` in `routes.rs:1497-1500`) and trivially forgeable. There is no Roughtime, RFC 3161, or Sigsum/Trillian anchor.

3. **`prev_hash` for the first entry is zeros.** A forger producing a chain from scratch starts here too — there is no genesis anchor.

4. **No per-server keypair binds the chain to the brain instance.** Even if the brain were considered the trust anchor (which would defeat "tamper-evident" against a malicious operator), there is no mechanism to verify that a given chain was produced by *this* `pi.ruv.io` deployment vs. a fork.

The code is correct as a hash chain (good test coverage, constant-time-not-required, SHAKE-256 is fine). It is mis-described in `CLAUDE.md` and in the comments at `witness.rs:1-4` ("tamper-evident log"). It is integrity-evident against accidental corruption only; it is **not** an audit log against any active adversary.

**Verdict: the witness-chain claim does not hold.** To make it true, sign each chain head with an ed25519 key held only by the brain (verifier already imports `ed25519_dalek` in `verify.rs`), publish the public key, and periodically anchor heads to an external transparency log.

## Recommendations (priority order)

1. **Add an Axum `middleware::from_fn` layer that requires `AuthenticatedContributor` for every route except an explicit allowlist** (`/v1/health`, `/v1/ready`, `/`, `/.well-known/*`, `/origin`, `/robots.txt`, `/sitemap.xml`, `/og-image.svg`, `/v1/status`, `/v1/challenge`). Default-deny instead of opt-in fixes 30+ missing-auth routes in one change.
2. **Move the `/internal/*` routes onto a separate `Router` bound to `127.0.0.1` (or a Unix domain socket).** Don't rely on documentation; enforce at the listener.
3. **Make `verify_system_key` fail-closed when `BRAIN_SYSTEM_KEY` is empty AND the build is not `cfg(debug_assertions)`.** Or require a `BRAIN_DEV_BYPASS=1` env flag and refuse to start without one of the two.
4. **Stop hard-coding `"mcp-sse-session"` in `handle_mcp_tool_call`.** Require the SSE client to provide an API key via the SSE handshake (e.g. `Authorization` on the `/sse` GET, or a query-string token validated identically), then propagate it to every loopback call. While doing this, also drop the `_api_key` JSON arg — keys belong in headers.
5. **Remove the `'anonymous'` default from `pi-brain/src/client.ts`.** Throw at construction time if no key is configured. Document `PI=` env in the README.
6. **Authenticate `pipeline_inject`, `pipeline_inject_batch`, `consciousness_compute`, and verify the Resend signature on `email_inbound`** (Svix verification: `npm i svix` server-side, validate `svix-id`, `svix-timestamp`, `svix-signature`).
7. **Validate Cloud Pub/Sub OIDC tokens in-process at `pipeline_pubsub_push`.** Don't rely on ambient IAM. Use `google-cloud-auth` ID-token verification against the configured push subscription's service account.
8. **Trust `X-Forwarded-For` only when the connecting peer matches the Cloud Run frontend range, or use Axum's `ConnectInfo` plus `tower-http`'s `RealIp` extractor configured for one trusted hop.**
9. **Differential privacy** — pick one path:
   - (a) Drop the DP claim from CLAUDE.md and the README; keep noise as a defense in depth, not a guarantee.
   - (b) Re-derive sigma with `sensitivity = 2 × clipping_norm`, wire up `PrivacyAccountant`, gate writes on `can_afford`, and **never store raw `title`/`content` alongside the noised embedding** when DP is enabled. Without (b)'s last clause, the math is irrelevant.
10. **Witness chain** — sign each chain head with an ed25519 key, publish the public key on `/v1/status`, periodically anchor heads to an external transparency log (Sigsum, or even a public Git repo). Without an external anchor, the chain is not auditable against a malicious operator.
11. **Raise `MIN_API_KEY_LEN` to 32 bytes** and reject low-entropy keys.
12. **Move `with_seed` on `DiffPrivacyEngine` behind `#[cfg(test)]`** (or rename to `with_seed_for_testing_only` and feature-gate it). Add a runtime check that panics if the seeded RNG is used while `dp_epsilon > 0`.
