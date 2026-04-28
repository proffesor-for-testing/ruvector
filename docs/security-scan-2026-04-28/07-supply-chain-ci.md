# Supply Chain & CI/CD Audit — RuVector

Date: 2026-04-28
Auditor: V3 QE Security Auditor
Scope: `.github/workflows/`, `.githooks/`, `scripts/`, `install.sh`, `Dockerfile*`, `.cargo/`, `.mcp.json`, `patches/`, `npm/packages/router-*`, `.gitmodules`
Mode: Read-only

## Summary

RuVector ships **5 NAPI router binaries** to npm, exposes a public `install.sh` invoked via `curl | bash`, and runs **38 GitHub Actions workflows** (24 of which trigger on `pull_request`). The publishing pipeline is well gated (publish jobs all require tags or `workflow_dispatch`), but the workflows have **systemic hardening gaps**: no top-level `permissions:` blocks anywhere, all third-party actions are pinned to mutable `@vN` tags or floating `@stable` refs (zero SHA pins), and PR-triggered build workflows shell-interpolate `github.event.inputs.*` without quoting in two workflows. None of these have produced an exploit, but a single compromise of `dtolnay/rust-toolchain`, `Swatinem/rust-cache`, or `napi-rs/cli@^2` would let an attacker swap binaries before they reach the publish step. The pre-built router binaries are **not in the source tree** (only skeleton package.json files at `npm/packages/router-*/`); they are produced by `build-router.yml` from source and uploaded as actions artifacts, then republished. They are **unsigned** (no Sigstore/cosign attestation). `install.sh` itself is acceptable for a dev-tool installer but executes `cargo install` and `npm install -g` from upstream registries with no checksum verification of the resulting binaries — the trust boundary is crates.io and npmjs.com, not RuVector.

## Critical: workflow-level RCE / token-leak risks

None confirmed. Specifically:
- **No `pull_request_target` triggers anywhere** (verified: zero matches across 38 workflows). This is the single most important class of CI RCE and it is absent — good.
- **NPM_TOKEN / CARGO_REGISTRY_TOKEN never reach pull_request runs.** All 9 publish workflows that read `secrets.NPM_TOKEN` or `secrets.CARGO_REGISTRY_TOKEN` are gated on `startsWith(github.ref, 'refs/tags/v')`, `inputs.publish == true`, or `github.event_name == 'workflow_dispatch'`. Examples: `build-router.yml:138`, `build-attention.yml:265`, `publish-all.yml:188,241`. Forks cannot reach the publish job.
- **`mirror-rulake.yml` uses a fine-grained PAT** (`secrets.RULAKE_MIRROR_PAT`, line 49) gated on `push` to `main` only — not exposed to PRs.

## High: unpinned actions, missing permissions, script-injection candidates

1. **Zero workflow has a top-level `permissions:` block.** `GITHUB_TOKEN` therefore defaults to whatever the repo setting is (write-all on classic, read-all on default). Some jobs declare `permissions:` (e.g. `docker-publish.yml:43`, `release.yml:443`), but the rest of the jobs in those same workflows inherit the default. Set `permissions: contents: read` at the top of every workflow and override per-job where needed.
2. **All third-party actions are tag-pinned, not SHA-pinned.** Inventory of distinct uses (line samples):
   - `actions/checkout@v4`, `actions/setup-node@v4`, `actions/upload-artifact@v4`, `actions/download-artifact@v4`, `actions/cache@v4`, `actions/github-script@v7`
   - `dtolnay/rust-toolchain@stable` (53 occurrences — floating branch, not even a tag)
   - `Swatinem/rust-cache@v2`, `taiki-e/install-action@v2` and `@cargo-llvm-cov`
   - `docker/login-action@v3`, `docker/build-push-action@v5`, `docker/setup-buildx-action@v3`, `docker/setup-qemu-action@v3`, `docker/metadata-action@v5`
   - `softprops/action-gh-release@v1` and `@v2` (mixed), `peter-evans/create-or-update-comment@v4`, `peter-evans/dockerhub-description@v4`
   - `addnab/docker-run-action@v3`, `benchmark-action/github-action-benchmark@v1`, `codecov/codecov-action@v4`, `rustsec/audit-check@v2`
   - `google-github-actions/auth@v2`, `google-github-actions/setup-gcloud@v2`, `google-github-actions/upload-cloud-storage@v2`
   - **Deprecated**: `actions-rs/toolchain@v1` (unmaintained since 2022), `actions-rust-lang/setup-rust-toolchain@v1`
   Pin every third-party action to a 40-char SHA per OpenSSF / GitHub Actions hardening guidance.
3. **Script-injection candidates** (`github.event.inputs.*` interpolated into `run:` shell with no quoting):
   - `edge-net-models.yml:67-68` — `MODELS="${{ github.event.inputs.models }}"` and `QUANT="${{ github.event.inputs.quantization }}"`
   - `docker-publish.yml:63` — `VERSION="${{ github.event.inputs.version }}"`
   - `release-rvf-cli.yml:131` — `echo "tag=${{ github.event.inputs.tag }}" >> "$GITHUB_OUTPUT"`
   These are `workflow_dispatch` only (attacker needs write access to trigger), so impact is low — but a hostile branch from a maintainer with weak 2FA could exfil secrets. Move to `env:` indirection (`env: TAG: ${{ github.event.inputs.tag }}` then use `"$TAG"`).
4. **`copilot-setup-steps.yml` is malformed** — every line after `on: workflow_call:` is indented progressively further with literal extra whitespace, suggesting it was hand-edited and would not parse if invoked. The file runs `npm install -g ruvector` from upstream registry without integrity checks.
5. **`npm install` without `--ignore-scripts`** in publish/release paths: `release.yml:71` (`npm ci`), `release.yml:528-534` (`npm install @ruvector/core`, `npm install -g @ruvector/cli`), `publish-all.yml:117,545,548`, `ruvllm-build.yml:167,229`, `agentic-synth-ci.yml:54,95,161,205,266,296`. Most build workflows correctly use `npm install --ignore-scripts --omit=optional --force` (good), but the publish/smoke-test paths execute postinstall scripts from the live registry mid-publish — a compromised transitive dep would run with the publish token in env.

## Medium: install script / container hardening

- **`install.sh`** (435 lines) is reasonable for a dev installer:
  - Uses `curl --proto '=https' --tlsv1.2 -sSf` for the rustup pipe (line 132) — good.
  - Does **not** verify any checksum/signature of the `cargo install ruvector-cli` output or `npm install -g ruvector` output. Trust delegated to crates.io and npmjs.com.
  - Uses `set +e` (line 7) deliberately to handle errors manually — fragile but not exploitable.
  - Suggests `sudo npm install -g` (line 208) — common but undesirable; recommend `npm config set prefix ~/.local`.
- **WASM-pack installation pattern repeated 3× inside CI workflows**: `curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh` (`publish-all.yml:163`, `build-attention.yml:173`, `release.yml:149`). `rustwasm.github.io` is a GitHub Pages site — recommend pinning a release tarball and verifying SHA256 instead.
- **Dockerfiles**: mixed quality.
  - **Good**: `examples/dragnes/Dockerfile` (multi-stage, non-root user `dragnes` UID 1001, HEALTHCHECK, `npm ci --ignore-scripts`).
  - **Acceptable**: `crates/ruvector-postgres/Dockerfile.prebuilt` (sets `POSTGRES_USER=ruvector`, has HEALTHCHECK, but inherits postgres image's user model).
  - **Poor**: `crates/mcp-brain-server/Dockerfile` — final stage `FROM debian:bookworm-slim` has **no `USER` directive** (runs as root) and **no `HEALTHCHECK`**. This is the binary that talks to the public `pi.ruv.io` brain.
  - `ui/ruvocal/Dockerfile` correctly drops to `USER user` after install.
- **`.dockerignore`** (root) is good — excludes `.git/`, `target/`, `node_modules/`, `.claude/`, `*.node`, `*.so`, `*.wasm`. No risk of secrets being baked into images from this repo's root build context.
- **`.cargo/config.toml`** is benign — no custom registry mirrors, only `git-fetch-with-cli = true` and a `RUST_MIN_STACK` env. No source-replacement attack surface.
- **`.mcp.json`** is benign — declares a single local stdio server (`aqe-mcp`) with no remote endpoints.
- **`.githooks/pre-commit`** runs `scripts/sync-lockfile.sh`, which calls `npm install --ignore-optional` (NOT `--ignore-scripts`) on every commit that touches `package.json`. A compromised dep would execute postinstall scripts on every developer machine that has installed the hook. Add `--ignore-scripts` to that npm call (`scripts/sync-lockfile.sh:30`).

## Pre-built binary review (router NAPI binaries — built or uploaded?)

**Verdict: built reproducibly in CI from source — not hand-uploaded. Unsigned.**

- The five `npm/packages/router-{darwin-arm64,darwin-x64,linux-arm64-gnu,linux-x64-gnu,win32-x64-msvc}/` directories in the source tree contain **only `package.json`** — no `.node` binaries committed to git (verified: each dir is exactly one file, ~580 bytes).
- Git log on those paths shows only metadata bumps (`9948c2f6 chore(router): bump to 0.1.30`, `314b40f0 feat: Add platform-specific npm packages for multi-platform support`). No `.node` blobs in history.
- `build-router.yml` builds each platform on the corresponding GitHub-hosted runner (`ubuntu-22.04`, `macos-14`, `windows-2022`) from the source `crates/ruvector-router-ffi/`, uploads to `actions/upload-artifact@v4`, then a single publish job downloads them and runs `npm publish`. This is the correct pattern.
- **However:** the binaries are not signed. There is no `cosign sign-blob` / `gh attestation` / Sigstore step. Downstream npm consumers cannot verify provenance beyond "the @ruvector publisher account had a valid NPM_TOKEN at publish time". Recommend adding `attestations: true` to `npm publish` (npm provenance, which uses Sigstore + GitHub OIDC) — this is a single-line CI change and would let consumers verify the binary was built by `build-router.yml` at a known commit. Same recommendation applies to all the other NAPI publish workflows (`build-attention`, `build-gnn`, `build-tiny-dancer`, `build-graph-node`, `build-rvf-node`, `build-diskann`, `build-graph-transformer`, `ruvllm-native`, `sona-napi`).
- Note: other prebuilt `.node` files do exist in subtrees (`npm/node_modules/...`, `npm/packages/rvf-node/*.node`) — these are local install artifacts and `node_modules` cache, not committed for distribution.

## Submodule / patch review

- **Submodule**: a single submodule, `examples/vectorvroom`, pointing at `https://github.com/shaal/VectorVroom.git`. Pinned to commit `4c2527b4526ccb8960cd13e3d9e1802d958dca60` (uninitialized in this checkout, indicated by leading `-`). Not floating — pinned to a SHA. Acceptable, but it is a third-party repo (`shaal/`, not `ruvnet/`). Confirm whoever owns `shaal/VectorVroom` is trusted; if not, vendor the example into the monorepo or fork it into `ruvnet/`.
- **Patches**: one entry, `patches/hnsw_rs/`, with full README explaining the patch (rationale: rand 0.9 → 0.8 for WASM compat, edition 2021 not 2024). The patch is applied via `[patch.crates-io]` in the workspace `Cargo.toml`, so every dependent crate uses this version, not the published one. Risk: if upstream `hnsw_rs` ships a CVE fix, RuVector won't get it automatically — the maintainers must manually rebase the patch. Add a `cargo audit` job that scans the workspace including patched crates (rustsec/audit-check@v2 is already used in some workflows, but should be in `ci.yml` against the patched build), and put a calendar reminder to re-sync the patch quarterly. The patch source itself is a fork of MIT/Apache-licensed code with `Changes.md` documenting modifications — clean.

## Recommendations (priority order)

1. **Add `permissions: contents: read` at the top of every workflow.** Override per-job where writes are needed (release, packages, pull-requests). Single highest-leverage hardening change.
2. **Enable npm provenance attestations on every NAPI publish workflow.** Add `--provenance` to each `npm publish` and `id-token: write` to the publish job's `permissions:`. Gives downstream users cryptographic proof the binary came from the documented CI workflow.
3. **SHA-pin all third-party actions.** Use `pin-github-actions` or `dependabot` with `groups` and `package-ecosystem: "github-actions"`. Prioritise `dtolnay/rust-toolchain@stable` (53 uses, floating branch).
4. **Move `github.event.inputs.*` into `env:` blocks** in `edge-net-models.yml`, `docker-publish.yml`, `release-rvf-cli.yml` to neutralise script-injection paths.
5. **Add `--ignore-scripts` to the local pre-commit npm call** (`scripts/sync-lockfile.sh:30`) and to the publish-time `npm install` calls in `release.yml` and `publish-all.yml`.
6. **Add `USER` and `HEALTHCHECK` to `crates/mcp-brain-server/Dockerfile`'s final stage**, since this binary is exposed to the network.
7. **Replace the `curl … | sh` wasm-pack pattern in CI** with a pinned release tarball + SHA256 verification (`taiki-e/install-action` already supports `wasm-pack`, which is what `ruvltra-tests.yml` uses — standardise on it).
8. **Migrate off `actions-rs/toolchain@v1`** wherever it appears (deprecated; use `dtolnay/rust-toolchain` SHA-pinned, or `actions-rust-lang/setup-rust-toolchain`).
9. **Verify ownership of `examples/vectorvroom` submodule** (`shaal/VectorVroom`) or vendor/fork it.

## Trust verdict for end users

- **`install.sh`**: trustworthy as a convenience wrapper. It does no privileged operations beyond what `rustup`/`cargo install`/`npm install -g` themselves do. The trust boundary is crates.io and npmjs.com, not this script. Users concerned about supply chain should run `cargo install ruvector-cli` and `npm install ruvector` directly with `--locked` / lockfile pinning.
- **Shipped NAPI router binaries**: trustworthy with one caveat — they are **built reproducibly in GitHub Actions from source on the matching runner OS** (no hand-uploads), but they are **unsigned** and lack npm provenance attestations. Until provenance is added, downstream verification reduces to "trust the @ruvector npm account hasn't been hijacked." Adding `--provenance` to the publish step (a one-line change) would close this gap.
