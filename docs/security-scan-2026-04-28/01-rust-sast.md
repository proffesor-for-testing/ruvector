# Rust SAST Findings

**Audit date:** 2026-04-28
**Scope:** `/workspaces/RuVector/crates/` (Rust source only, read-only)
**Tooling used:** `rg` pattern scans (cargo-audit / cargo-geiger were NOT installed in the sandbox; recommend running both before release).
**Method:** Pattern-based static review focused on critical sub-trees: `mcp-brain-server`, `mcp-brain`, `mcp-gate`, `cognitum-gate-kernel`, `ruvector-router-ffi`, `ruvector-temporal-tensor` (FFI), `rvAgent`, `rvf-*`, `ruvector-cli`. Pure-math crates (sparsifier, mincut, solver) skipped per scope.

## Summary (counts by severity)

| Severity | Count |
|---|---|
| **Critical** | 4 |
| **High** | 5 |
| **Medium** | 7 |
| **Low** | 6 |
| **Notes / FP candidates** | 4 |

Total findings: **22 issues** + 4 informational notes.

---

## Critical findings

### C-1. Auth bypass: any 8-character bearer token is accepted as a valid contributor
- **File / line:** `/workspaces/RuVector/crates/mcp-brain-server/src/auth.rs:99`
- **Snippet:**
  ```rust
  if api_key.len() < MIN_API_KEY_LEN || api_key.len() > 256 {
      return Err((StatusCode::UNAUTHORIZED, "Invalid API key"));
  }
  if let Some(ref system_key) = *SYSTEM_KEY {
      if api_key.as_bytes().ct_eq(system_key.as_bytes()).into() { ... }
  }
  Ok(Self::from_api_key(api_key))   // <-- ALWAYS returns Ok
  ```
- **Exploit:** `Authorization: Bearer aaaaaaaa` (8 bytes) deterministically derives a SHAKE-256 pseudonym and is accepted as a "contributor". An attacker can register memories, vote, submit deltas, and consume contributor rate-limit budget under any pseudonym they invent. There is no API key registry / database of valid keys — the only checks are length and SHAKE hash derivation.
- **Fix:** Reject when neither the system key matches nor the key is registered in Firestore (`state.store.api_key_exists(api_key_hash)`); return `401` instead of `Ok(from_api_key(...))`.

### C-2. `verify_system_key` fails-OPEN when `BRAIN_SYSTEM_KEY` env var is unset
- **File / line:** `/workspaces/RuVector/crates/mcp-brain-server/src/routes.rs:8038-8050`
- **Callers (7 internal endpoints):** lines `6532, 6568, 6623, 6679, 6726, 6765, 7003` — all internal/admin email + queue endpoints (around `/v1/internal/*`, notifier ops).
- **Snippet:**
  ```rust
  fn verify_system_key(headers: &HeaderMap) -> Result<...> {
      let system_key = std::env::var("BRAIN_SYSTEM_KEY").unwrap_or_default();
      if system_key.is_empty() { return Ok(()); }   // <-- bypass
      ...
  }
  ```
- **Exploit:** Any deployment (Docker image, Cloud Run revision, k8s pod) where the `BRAIN_SYSTEM_KEY` secret was forgotten silently exposes 7 admin/internal endpoints to the world. Comment says "dev mode" but this is enforced in production code with no guard.
- **Fix:** When `system_key` is empty, return `503 SERVICE_UNAVAILABLE "system key not configured"` and refuse to start the server in production builds (gate on `cfg!(debug_assertions)` or an explicit `ALLOW_INSECURE_DEV` env flag).

### C-3. Command injection via single-quote escape in sandbox file ops (`sh -c`)
- **Files / lines:**
  - `/workspaces/RuVector/crates/rvAgent/rvagent-backends/src/sandbox.rs:123` — ``self.execute_sync(&format!("cat -n '{}'", file_path), None);``
  - `/workspaces/RuVector/crates/rvAgent/rvagent-backends/src/sandbox.rs:142` — ``&format!("ls -la --time-style=full-iso '{}' 2>/dev/null", path),``
  - Implementation: `sandbox.rs:244-250` — `Command::new("sh").arg("-c").arg(command)`
- **Exploit:** `file_path = "x'; rm -rf $HOME; echo '"` produces ``cat -n 'x'; rm -rf $HOME; echo ''`` — arbitrary shell execution inside the sandbox root with the agent's privileges. Although `env_clear()` and `PATH=/usr/bin:/bin` are set, `sh`, `rm`, `curl`, `nc` etc. are still on PATH.
- **Fix:** Stop using `sh -c` for file ops. Use `Command::new("cat").arg(file_path)` directly so the path becomes a single argv entry. Better yet, perform `cat`/`ls` in pure Rust (`std::fs::read_to_string`, `std::fs::read_dir`).

### C-4. Direct user-input shell execution in agent `execute()` tool
- **File / line:** `/workspaces/RuVector/crates/rvAgent/rvagent-cli/src/app.rs:332-340`
- **Snippet:**
  ```rust
  fn execute(&self, command: &str, timeout_secs: u32) -> ... {
      use std::process::Command;
      let output = Command::new("sh")
          .arg("-c")
          .arg(command)         // <-- raw string from agent tool call
          .current_dir(&self.cwd)
          .output() ...
  ```
- **Exploit:** This is the LLM tool surface. Any prompt-injection in indexed memories or web pages that reaches this tool yields full RCE on the host (no chroot, no env scrubbing, runs in CWD). Comment "timeout handled at a higher level if needed" indicates the timeout is also unenforced.
- **Fix:** This appears intentional for an agent CLI, but it must be (1) gated behind explicit user approval per call, (2) sandboxed (e.g., reuse the `rvf` microVM crate sitting next door), (3) timeout-enforced via `tokio::time::timeout`, and (4) excluded entirely from production / hosted deployments.

---

## High findings

### H-1. Data-race UB via `static mut` global accessed from FFI
- **Files / lines:**
  - `/workspaces/RuVector/crates/ruvector-temporal-tensor/src/store_ffi.rs:200` — `static mut STORE_STATE: Option<StoreState> = None;`
  - Read/write sites: `216, 217, 223, 330, 562`
  - `/workspaces/RuVector/crates/ruvector-temporal-tensor/src/ffi.rs:13` — `static mut STORE: Option<Vec<Option<TemporalTensorCompressor>>> = None;`
- **Snippet:**
  ```rust
  static mut STORE_STATE: Option<StoreState> = None;
  ...
  unsafe {
      if STORE_STATE.is_none() { STORE_STATE = Some(StoreState { ... }); }
      f(STORE_STATE.as_mut().unwrap())
  }
  ```
- **Exploit:** Although intended for single-threaded WASM, when these `cdylib`s are linked into Node (NAPI) or used as a shared library in a multi-threaded host, two threads calling `tts_init` / `ttc_push_frame` concurrently produce a data race — undefined behaviour, possible double-free of `StoreState` contents.
- **Fix:** Wrap in `parking_lot::Mutex<Option<StoreState>>` inside a `LazyLock`. If the API is only ever single-threaded, document and `#[cfg(target_arch = "wasm32")]`-gate the module so the unsound version cannot be compiled for non-WASM targets.

### H-2. FFI `ttc_dealloc` reconstructs `Vec` with wrong `(len, cap)` — UB and memory corruption
- **File / line:** `/workspaces/RuVector/crates/ruvector-temporal-tensor/src/ffi.rs:243-249`
- **Snippet:**
  ```rust
  pub extern "C" fn ttc_dealloc(ptr: u32, cap: u32) {
      ...
      unsafe {
          let _ = Vec::<u8>::from_raw_parts(ptr as *mut u8, 0, cap as usize);
      }
  }
  ```
  And the matching alloc at line 233:
  ```rust
  let mut v: Vec<u8> = Vec::with_capacity(size as usize);
  let p = v.as_mut_ptr();
  std::mem::forget(v);
  ```
- **Exploit:** `Vec::with_capacity` may over-allocate (e.g. capacity rounded up to a multiple of 8 / power of two), so `cap` passed back by JS WASM glue rarely matches the actual capacity. `from_raw_parts` *requires* the same `cap` the Vec was created with — mismatch is **undefined behaviour** per the docs. On native targets this can cause heap corruption; on WASM it leaks memory but can also free regions still in use if the host caches old pointers. Additionally, `ptr as u32` truncates 64-bit pointers (fine on wasm32, broken if compiled for native arch).
- **Fix:** Track allocations in a side-table `HashMap<*mut u8, Layout>` and use `std::alloc::dealloc` with the recorded `Layout`. Or always allocate with an explicit `Layout` via `alloc()` and free via `dealloc()`.

### H-3. Per-IP rate limiting trivially bypassed via spoofed `X-Forwarded-For`
- **File / line:** `/workspaces/RuVector/crates/mcp-brain-server/src/routes.rs:33-43, 1435-1436, 2279-2282`
- **Snippet:**
  ```rust
  fn extract_client_ip(headers: &HeaderMap) -> String {
      headers.get("x-forwarded-for")
          .and_then(|v| v.to_str().ok())
          .and_then(|v| v.split(',').next())
          .map(|s| s.trim().to_string())
          .unwrap_or_else(|| "unknown".to_string())
  }
  ```
- **Exploit:** When the server is accessed directly (any deployment without a trusted proxy in front, e.g. local Docker, a misconfigured Cloud Run revision with `--allow-unauthenticated`, or a developer port-forward), the attacker simply sends `X-Forwarded-For: 1.2.3.<random>` per request and bypasses `check_ip_write` / `check_ip_vote`. Combined with **C-1**, this enables unbounded vote stuffing / Sybil attacks on the reputation system.
- **Fix:** Trust XFF only when the immediate connection IP is in a hard-coded allow-list of trusted proxies (e.g. Google Cloud LB CIDRs). Otherwise use `ConnectInfo<SocketAddr>` from axum.

### H-4. Path-traversal in blob store via attacker-controlled `content_hash`
- **File / line:** `/workspaces/RuVector/crates/mcp-brain-server/src/bin/local.rs:705-722`
- **Snippet:**
  ```rust
  fn blob_write(blob_dir: &str, hash: &str, content: &str) {
      if hash.len() < 4 { return; }
      let dir = format!("{}/{}", blob_dir, &hash[..2]);
      let _ = std::fs::create_dir_all(&dir);
      let path = format!("{}/{}", dir, &hash[2..]);
      let _ = std::fs::write(path, content);
  }
  fn blob_read(blob_dir: &str, hash: &str) -> Option<String> {
      ... let path = format!("{}/{}/{}", blob_dir, &hash[..2], &hash[2..]);
      std::fs::read_to_string(path).ok()
  }
  ```
- **Exploit:** In the writer, `hash` is computed by `content_hash()` (SHA-256 hex) so the writer side is safe. In the reader, however, callers (e.g. lines `857, 1262`) pass `hash` strings that came in from the database — and the database was populated by external API calls. The `add_evidence` / `share` paths recompute the hash from supplied content, but a malicious operator with DB access (or a future code path that loads `content_hash` from user JSON without re-hashing) could insert `"../../etc/passwd"`. **Needs verification of every caller**; the function itself does no validation (no check that `hash` is hex, no canonicalisation against `blob_dir`).
- **Fix:** Validate `hash.chars().all(|c| c.is_ascii_hexdigit()) && hash.len() == 64` at the top of both functions. Optionally `Path::canonicalize` and assert `path.starts_with(blob_dir)` after construction.

### H-5. `unwrap()` on `serde_json::to_value` in the partition handler (DoS panic on serializer error)
- **File / line:** `/workspaces/RuVector/crates/mcp-brain-server/src/routes.rs:2636, 2638, 2672, 2674, 194`
- **Snippet:**
  ```rust
  return Ok(Json(serde_json::to_value(compact).unwrap()));
  ...
  return Ok(Json(serde_json::to_value(cached).unwrap()));
  ```
- **Exploit:** `serde_json::to_value` returns `Err` for non-string-key maps and for types whose `Serialize` impl errors (e.g. invalid floats `NaN`/`±Inf` if `arbitrary_precision` is on; or our own custom serializers). A crafted partition cluster containing `f64::NAN` would panic the worker thread. Tokio handles thread panics with abort on some configs; at minimum it terminates the request future ungracefully and pollutes logs.
- **Fix:** Replace `.unwrap()` with `.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!(...)))?`. Also `.expect("valid DP parameters")` at line 194 should be a typed error returned to the caller.

---

## Medium findings

### M-1. Unencrypted `http://` Brain URL fallback for system-to-system calls
- **File / line:** `/workspaces/RuVector/crates/mcp-brain-server/src/bin/local.rs:884`
  ```rust
  .unwrap_or_else(|_| "http://127.0.0.1:9877".to_string());
  ```
  And `/workspaces/RuVector/crates/mcp-brain-server/src/routes.rs:5909` — `let base = format!("http://127.0.0.1:{port}");`
- **Risk:** Localhost is fine; documented for completeness because the env var (`BRAIN_URL`) can be overridden by an attacker with environment access to point at an `http://` external host, leaking API keys passed in the `Authorization: Bearer` header (see `gist.rs:762`).
- **Fix:** Hard-reject non-`https` schemes when `BRAIN_URL` is not `localhost`/`127.0.0.1`.

### M-2. `recv` of attacker-controlled JSON without upper bound on session sender
- **File / line:** `/workspaces/RuVector/crates/mcp-brain-server/src/routes.rs:5343`
- **Snippet:** `let request: serde_json::Value = match serde_json::from_str(&body) { ... }`
- **Risk:** The route does *not* sit behind the global `RequestBodyLimitLayer(2_097_152)` if it is mounted as a sub-router elsewhere — needs verification. Also `serde_json::Value` deserialisation has quadratic time on extremely deep nesting (stack overflow on `[[[[[[...]]]]]]` ~10k deep).
- **Fix:** Use `serde_json::from_slice` with an explicit max-depth via `serde_json::Deserializer::from_slice(...).disable_recursion_limit()` *off* (default 128 is fine) and confirm `RequestBodyLimitLayer` covers SSE POSTs.

### M-3. CORS allow-list parsing silently drops malformed entries (header injection candidate)
- **File / line:** `/workspaces/RuVector/crates/mcp-brain-server/src/routes.rs:413-422`
- **Snippet:** `.split(',').filter_map(|s| s.trim().parse::<HeaderValue>().ok()).collect();`
- **Risk:** Operators may add a stray space or non-ASCII character to `CORS_ORIGINS` and have an entry silently disappear without log. Not directly exploitable; misconfiguration risk only.
- **Fix:** Log a `warn!` on each rejected entry; refuse to start if the parsed list is empty after a non-empty input.

### M-4. `current_dir(&self.cwd)` without canonicalising — TOCTOU
- **File / line:** `/workspaces/RuVector/crates/rvAgent/rvagent-cli/src/app.rs:336`
- **Risk:** If `self.cwd` is a symlink that the agent itself can rewrite, the child shell ends up in a different directory than intended.
- **Fix:** `cwd.canonicalize()?` once at session start; re-validate before each `Command` spawn.

### M-5. `MD5` used for cache keys — collision risk for cache poisoning
- **Files / lines:**
  - `/workspaces/RuVector/crates/ruvllm/src/context/working_memory.rs:394, 423`
  - `/workspaces/RuVector/crates/ruvllm/src/context/semantic_cache.rs:184, 285`
- **Snippet:** `let input_hash = format!("{:x}", md5::compute(input));`
- **Risk:** If the cache key is ever cross-tenant or used to look up stored prompts/replies, MD5 collisions could let one user's input fetch another user's cached response. For an in-process single-tenant cache, severity is low — flagged Medium because the crate name suggests an LLM cache and these are typically multi-user.
- **Fix:** Use SHA-256 (already a dependency in `verify.rs`); switch to BLAKE3 if perf matters.

### M-6. `eprintln!` of bind address may leak host info to logs scraped by attackers
- **File / line:** `/workspaces/RuVector/crates/mcp-brain-server/src/bin/local.rs:548`
- **Risk:** Trivial; informational leak only.
- **Fix:** Demote to `tracing::info!` so log levels can hide it.

### M-7. `ttc_alloc` returns the raw pointer of a Vec via `mem::forget`, but the caller has no way to know the *capacity actually allocated*
- **File / line:** `/workspaces/RuVector/crates/ruvector-temporal-tensor/src/ffi.rs:228-239`
- **Risk:** Tied to **H-2**. Even ignoring the dealloc bug, the allocator may give back fewer bytes than requested if `Vec` over-allocates and the caller writes `size` bytes — that part is fine — but if it ever writes `cap` bytes (taken from a returned size), it would write past the allocation.
- **Fix:** Return both `ptr` *and* the actual capacity (`v.capacity() as u32`) via two out-params.

---

## Low findings

### L-1. `unwrap_or_default()` swallows TLS/IO errors in `verify_system_key` header read
- `routes.rs:8048` — `headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("")`. Returns `""` on non-UTF-8 header → constant-time compare with system key never matches → 401. Defensive, but opaque to the operator. **Fix:** log a warn.

### L-2. Many `unwrap()`s in `routes.rs` (5 instances) and `local.rs` (`.unwrap()` on `/proc/loadavg`, line 966)
- `local.rs:966` — `std::fs::read_to_string("/proc/loadavg").unwrap_or_default()` is fine (default), but other call-sites should be audited. Risk of panic in containerised environments without `/proc`.

### L-3. `rand::thread_rng()` used for `ed25519_dalek::SigningKey::generate` in tests
- `verify.rs:375` — test-only, not exploitable. Not a real finding.

### L-4. `Connection::open(&path)?` then `execute_batch("PRAGMA journal_mode=WAL; ...")` without IF NOT EXISTS guards
- `bin/local.rs:716-717` — fine semantically but `journal_mode=WAL` is a *file format* change. If two processes open the same DB with different journal modes, one will silently lose changes.

### L-5. `tokio::process::Command::new("nvidia-smi")` runs without explicit absolute path
- `bin/local.rs:949` — relies on `$PATH`. If a hostile entry is ahead of `/usr/bin` in PATH, attacker-controlled `nvidia-smi` runs. Standard mitigation: use `which` once at startup, then absolute path.

### L-6. `tokio::process::Command::new("gcloud")` similarly relies on PATH
- `crates/ruvector-kalshi/src/secrets.rs:153` and `crates/ruvector-kalshi/examples/paper_trade.rs:55`. Same fix.

---

## Notes / false-positive candidates

- **`unsafe` in `crates/cognitum-gate-kernel/src/{shard,delta}.rs`** — every block has a `// SAFETY:` comment and is preceded by an explicit bounds check or a `#[repr(C)] union` tag check. Reviewed; appears sound. Not a finding.
- **`unsafe` in `crates/ruvector-core/src/{arena,cache_optimized,simd_intrinsics}.rs`** — arena/cache slices use `from_raw_parts` after the type's own bookkeeping (`self.len`, `self.dimensions`). SIMD `transmute([f32;8]; __m256)` is the canonical pattern. Reviewed; appears sound.
- **`Command::new("docker"|"clang"|"qemu-system-x86_64")` in `rvf-kernel/docker.rs`, `rvf-ebpf/lib.rs`, `rvf-launch/qemu.rs`** — all arguments are crate-internal constants or paths from the operator-supplied config. No user-input injection point. Not a finding.
- **`Command::new("psql")` in `crates/ruvector-cli/src/cli/hooks.rs:1142`** — `pg_url` and `schema_sql` are constants from the local install. CLI tool, operator-trusted. Not a finding.

---

## Recommended next steps

1. **Install `cargo-audit`** and run `cargo audit --json` against the workspace `Cargo.lock` — this audit did NOT cover known CVEs in transitive deps.
2. **Install `cargo-geiger`** and run on the FFI/CLI/server crates to get a numeric `unsafe` density score per crate.
3. **Run `clippy --all-targets -- -D clippy::indexing_slicing -D clippy::unwrap_used -D clippy::expect_used`** — will flag many of the L-2 / L-3 cases automatically.
4. **Manually verify H-4** — trace every caller of `blob_read` to confirm `hash` is always a freshly-computed SHA-256 and never directly attacker-controlled.
5. **Critical fixes (C-1, C-2) should ship before the next public release** — they together turn the brain server into an open-write database.
