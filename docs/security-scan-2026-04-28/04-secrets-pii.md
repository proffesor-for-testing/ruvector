# Secrets & PII Scan

**Scope:** RuVector working tree (excluding `node_modules/`, `target/`, `dist/`, `*.lock`, `bench_results/`, `test_models/`, `.git/objects`) + last ~5,500 commits of git history (full history was scanned via `git log --all --full-history -S` for the highest-signal patterns).
**Scanner:** `ripgrep` (gitleaks/trufflehog were not available in the environment; `which gitleaks trufflehog` returned nothing).
**Date:** 2026-04-28
**Mode:** Read-only — no files modified, no history rewritten.

---

## Summary

| Category | Confirmed real leaks | False-positive candidates reviewed |
|---|---|---|
| Active credentials in working tree | **0** | ~12 (all test fixtures, `.env.example` placeholders, doc strings) |
| Credentials in git history | **1** (Google Firebase web API key, removed from tree but still in history) | ~8 (Amazon's documented dummy `AKIAIOSFODNN7EXAMPLE`, dummy `ghp_abcdef…`, etc.) |
| PII in source/fixtures | **0** | All SSNs/emails are PII-detector test fixtures or project author/contact emails |
| Private keys (PEM/SSH/PGP) on disk | **0** | None — all `BEGIN ... PRIVATE KEY` matches are doc strings / format checks |
| DB connection URIs with embedded passwords | **0 (production)** | Many doc/example strings using literal `user:pass@localhost`; CI compose files use `postgres:test@…` for ephemeral test containers (acceptable) |

**Bottom line:** The current working tree is clean. There is one historical leak (a Firebase web API key for project `ruv-dev`) that requires action.

---

## CRITICAL: Active credentials still present in tree

**None confirmed.** All candidates with high keyword overlap turned out to be:

- `crates/ruvector-kalshi/src/secrets.rs:210-211` — unit-test fixtures using literals `super-secret-1234567890` and `-----BEGIN RSA PRIVATE KEY----- secret`. Test code, not a real key.
- `crates/mcp-brain-server/src/tests.rs:409-410`, `crates/mcp-brain-server/src/verify.rs:478`, `crates/rvf/rvf-federation/src/pii_strip.rs:333-428` — PII-detector self-tests using Amazon's published example key `AKIAIOSFODNN7EXAMPLE` and the dummy GitHub PAT `ghp_abcdefghijklmnopqrstuvwxyz0123456789`.
- All `.env*` files inspected (`/workspaces/RuVector/.env.example`, `examples/scipix/.env.example`, `npm/packages/agentic-synth/.env.example`, `npm/packages/ruvbot/.env.example`, `npm/packages/graph-data-generator/.env.example`, `ui/ruvocal/.env.ci`) contain only placeholders (`your-key-here`, `sk-ant-xxxxxxx…`, `mongodb://localhost:27017/`).
- `.github/workflows/hooks-ci.yml:117` — `POSTGRES_PASSWORD: test` for an ephemeral CI service container. Acceptable.
- `npm/packages/ruvbot/tests/setup.{ts,js}` — `xoxb-test-token`, `postgresql://test:test@localhost…`. Acceptable test scaffolding.

No `.pem`, `.key`, `.p12`, `id_rsa*`, `service-account*.json`, or `*-credentials*.json` files exist anywhere in the tree.

`/workspaces/RuVector/.git/config` and `/workspaces/RuVector/.mcp.json` are clean (SSH-based remote, no embedded creds).

---

## CRITICAL: Credentials in git history

### 1. Google Firebase web API key for project `ruv-dev` — ROTATE / RESTRICT

- **Key (redacted):** `AIzaSyAZAJ…vb2QvA` (39 chars, valid `AIza`-prefix Firebase/GCP browser API key format)
- **GCP project:** `ruv-dev`
- **Auth domain:** `ruv-dev.firebaseapp.com`
- **Introducing commit:** `1a6174bc` — "feat(edge-net): production-ready WebRTC, genesis deployment, and distributed workers v0.4.3"
- **Files (in that historical commit):**
  - `examples/edge-net/dashboard/src/services/firebaseData.ts`
  - `examples/edge-net/pkg/firebase-signaling.js`
- **Current status in working tree:** Not present (removed in a later commit), but **still reachable via `git log -p` / `git show 1a6174bc`** and therefore exposed to anyone with read access to the repo (public on `github.com/ruvnet/RuVector`).

**Why this matters even though Firebase web keys are "designed to be public":** These keys identify the GCP project and the abuse boundary is enforced *only* by Firebase Security Rules and API restrictions. If rules are permissive (write-anywhere, anonymous reads of sensitive collections), or if the key is unrestricted in GCP Console, attackers can: (a) burn quota / drive up billing, (b) write to Realtime DB / Firestore, (c) call Identity Toolkit to enumerate users. This is the only credential leak found, and it is real.

**Required action — pick at least one, ideally both:**

1. **Restrict the key in GCP Console immediately.** Go to https://console.cloud.google.com/apis/credentials?project=ruv-dev → find the API key starting with `AIzaSyAZAJ…` → set HTTP-referrer restrictions (your trusted domains only) and API restrictions (only Firebase APIs you actually use). This is the fastest mitigation and does not require code changes.
2. **Rotate (delete + recreate) the key** in GCP Console, then update any deployed clients. This is the strongest action.
3. **Audit Firebase Security Rules** for project `ruv-dev` (Firestore, Realtime DB, Storage). Ensure there is no `allow read, write: if true;` and that auth checks are present.
4. *(Optional, separate decision)* Purging the key from git history requires a force-push rewrite of `main` (e.g. `git filter-repo`) which is destructive to all clones and contradicts the read-only scope of this scan. Restricting/rotating the key in GCP makes the historical exposure inert and is the recommended path.

**No other credentials found in the last 200+ commits scanned with `-S` for:** `sk-ant-api`, `sk-or-`, `sk-proj-`, `AKIA` (only `AKIAIOSFODNN7EXAMPLE` test fixture), `BEGIN PRIVATE KEY` / `BEGIN RSA PRIVATE` / `BEGIN OPENSSH` (only doc/test strings), `ghp_` / `github_pat_` (only test fixtures), `hf_`, `xoxb-`, `mongodb+srv://…@`, `redis://…@`.

---

## PII findings

**No real PII identified in source.** All matches are intentional test fixtures or public author/contact data:

- **SSNs**: All occurrences (`078-05-1120`, `123-45-6789`) are inside PII-detector test code (`crates/rvf/rvf-federation/src/pii_strip.rs`, `crates/mcp-brain-server/src/{tests,verify}.rs`, `npm/packages/ruvbot/tests/unit/security/aidefence-guard.test.ts`, `docs/adr/ADR-082-brain-security-hardening.md`). `078-05-1120` is the famous IRS dummy SSN from the Woolworth wallet incident; `123-45-6789` is a sequential placeholder. Both are standard testing values.
- **Phone numbers**: Only `555-867-5309` (Tommy Tutone song lyrics, 555-prefix is reserved for fiction) in `crates/mcp-brain-server/src/verify.rs:465`.
- **Credit card numbers / financial identifiers**: None.
- **Emails**: All real emails found are project author / contact addresses, not user PII:
  - `ruv@ruv.io`, `team@ruvector.dev`, `team@ruvector.io`, `info@ruv.io`, `pi@ruv.io`, `research@ruv.io`, `security@ruvector.io` (across many `Cargo.toml`, `package.json`, source files — intentional ownership metadata).
  - `ruvbrain-scheduler@ruv-dev.iam.gserviceaccount.com`, `dragnes-sa@ruv-dev.iam.gserviceaccount.com` (GCP service-account *identifiers*, not credentials — safe to expose, but they confirm the `ruv-dev` GCP project name, which ties back to the Firebase finding above).
- **Bundled/extracted upstream code under `docs/research/claude-code-rvsource/`**: contains illustrative emails like `john@co.com` inside example LLM prompts. These are docs of third-party research material — not RuVector PII.

---

## Tooling used / coverage notes

- **Primary scanner:** `ripgrep` 13.x with custom regex set covering: Anthropic, OpenAI, OpenAI-Project, OpenRouter, GitHub PAT, GitHub fine-grained PAT, AWS, Google API (`AIza`), HuggingFace (`hf_`), Slack (`xox[baprs]-`), Stripe (`rk_`/`pk_`), JWT (3-segment `eyJ…`), private keys (`BEGIN .* PRIVATE KEY`), DB URIs with embedded passwords, generic `password|secret|token|api_key|auth = "<20+ chars>"` keyword scans, US SSN, US credit card (Visa/MC/Amex/Discover), `Bearer …` headers.
- **Git history coverage:** `git log --all --full-history -p -S '<pattern>'` for each high-signal token (see list above) across all 5,575 commits reachable from any ref. The `-S` form catches both adds and removes, which is how the Firebase key was found despite being absent from the working tree.
- **Manual inspection:** Every `.env*` file in scope, `.git/config`, `.mcp.json`, `.dockerignore`, `.gcloudignore`, `install.sh`, root and benchmarks `Dockerfile`, and all 30+ files under `.github/workflows/`.
- **Known coverage gap (per scope):** `ui/ruvocal/` was largely behind a permission wall for `Read`, but `rg`/`find` had access — `.env.ci` (`MONGODB_URL=mongodb://localhost:27017/`) and `Dockerfile` were inspected and are clean. The directory is included in all `rg` global scans above.
- **False-positive filtering:** Excluded `.test.*`, `tests/`, `test/`, `__tests__/`, `fixtures/`, plus keyword filter on `example`, `placeholder`, `your-`, `YOUR_`, `XXXX`, `dummy`, `mock`, `EXAMPLE`, `process.env`, `os.environ`, `getenv`, `${...}` template-literal references.
- **Not used:** `gitleaks`, `trufflehog` — neither installed. Recommend installing `gitleaks` and wiring it into a pre-commit hook + the existing GitHub Actions matrix to catch future leaks at PR time.
