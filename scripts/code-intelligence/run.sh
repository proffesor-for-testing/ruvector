#!/usr/bin/env bash
# =============================================================================
# Code Intelligence Scanner — Orchestrator
#
# Entry point for QE agents and CI to invoke incremental code intelligence
# indexing. Runs the indexer, stores results in AQE memory, and prints a
# machine-readable summary.
#
# Usage:
#   ./scripts/code-intelligence/run.sh [OPTIONS]
#
# Options:
#   --full                Full re-index (ignore incremental state)
#   --incremental         Only index files changed since last scan (default)
#   --scope <path>        Scope to a sub-project (e.g., crates/ruvector-core)
#   --lang <lang,...>     Filter languages (ts,js,rs,py,go,java)
#   --max-files <n>       Max files per sub-project (default: 5000)
#   --dry-run             Index but don't store to AQE memory
#   --verbose             Verbose output
#   --skip-store          Run indexer only, skip AQE memory store
#   --query <text>        After indexing, search the index for <text>
#
# Examples:
#   # Incremental scan of entire monorepo
#   ./scripts/code-intelligence/run.sh
#
#   # Full re-index of Rust crates only
#   ./scripts/code-intelligence/run.sh --full --scope crates --lang rs
#
#   # Scan npm packages, then search for "attention"
#   ./scripts/code-intelligence/run.sh --scope npm/packages --query "attention"
#
#   # CI mode: full index, verbose, store to AQE
#   ./scripts/code-intelligence/run.sh --full --verbose
#
# For QE agents:
#   Invoke via Bash tool:
#     bash scripts/code-intelligence/run.sh --incremental
#   Then query results:
#     bash scripts/code-intelligence/run.sh --query "auth middleware"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/.agentic-qe/data/code-intel"
INDEX_FILE="$OUTPUT_DIR/index.json"

# Defaults
MODE="--incremental"
SCOPE=""
LANG=""
MAX_FILES=""
VERBOSE=""
DRY_RUN=""
SKIP_STORE=""
QUERY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      MODE=""
      shift
      ;;
    --incremental)
      MODE="--incremental"
      shift
      ;;
    --scope)
      SCOPE="--scope $2"
      shift 2
      ;;
    --lang)
      LANG="--lang $2"
      shift 2
      ;;
    --max-files)
      MAX_FILES="--max-files $2"
      shift 2
      ;;
    --verbose)
      VERBOSE="--verbose"
      shift
      ;;
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --skip-store)
      SKIP_STORE="true"
      shift
      ;;
    --query)
      QUERY="$2"
      shift 2
      ;;
    -h|--help)
      head -45 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
      exit 0
      ;;
    *)
      echo "[code-intel] Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Query mode: search existing index without re-indexing
# ---------------------------------------------------------------------------

if [[ -n "$QUERY" && -f "$INDEX_FILE" && -z "$SCOPE" && -z "$LANG" ]]; then
  echo "[code-intel] Searching index for: $QUERY" >&2

  # Search entities by name
  ENTITY_MATCHES=$(node -e "
    const data = require('$INDEX_FILE');
    const q = '$QUERY'.toLowerCase();
    const matches = data.entities
      .filter(e => e.name.toLowerCase().includes(q) || (e.file && e.file.toLowerCase().includes(q)))
      .slice(0, 20)
      .map(e => ({ name: e.name, type: e.type, file: e.file, line: e.line, project: e.project }));
    console.log(JSON.stringify({ query: '$QUERY', matches, total: matches.length }, null, 2));
  " 2>/dev/null || echo '{"query":"$QUERY","matches":[],"total":0}')

  echo "$ENTITY_MATCHES"
  exit 0
fi

# ---------------------------------------------------------------------------
# Index mode
# ---------------------------------------------------------------------------

echo "[code-intel] Starting code intelligence scan..." >&2
echo "[code-intel] Root: $PROJECT_ROOT" >&2
echo "[code-intel] Mode: ${MODE:-full}" >&2

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run indexer
export PROJECT_ROOT
INDEX_ARGS="$MODE $SCOPE $LANG $MAX_FILES $VERBOSE --output $OUTPUT_DIR"

echo "[code-intel] Running indexer..." >&2
INDEXER_RESULT=$(node "$SCRIPT_DIR/index.mjs" $INDEX_ARGS)
INDEXER_EXIT=$?

if [[ $INDEXER_EXIT -ne 0 ]]; then
  echo "[code-intel] Indexer failed with exit code $INDEXER_EXIT" >&2
  exit $INDEXER_EXIT
fi

# Verify output exists
if [[ ! -f "$INDEX_FILE" ]]; then
  echo "[code-intel] Error: Index file not created at $INDEX_FILE" >&2
  exit 1
fi

# Print index summary
FILE_COUNT=$(node -p "require('$INDEX_FILE').summary.totalFiles" 2>/dev/null || echo "?")
ENTITY_COUNT=$(node -p "require('$INDEX_FILE').summary.totalEntities" 2>/dev/null || echo "?")
EDGE_COUNT=$(node -p "require('$INDEX_FILE').summary.totalEdges" 2>/dev/null || echo "?")
echo "[code-intel] Indexed: $FILE_COUNT files, $ENTITY_COUNT entities, $EDGE_COUNT edges" >&2

# ---------------------------------------------------------------------------
# Store to AQE memory
# ---------------------------------------------------------------------------

if [[ -z "$SKIP_STORE" ]]; then
  echo "[code-intel] Storing results in AQE memory..." >&2
  STORE_ARGS=""
  [[ -n "$DRY_RUN" ]] && STORE_ARGS="$STORE_ARGS --dry-run"

  node "$SCRIPT_DIR/store-to-aqe.mjs" --input "$INDEX_FILE" $STORE_ARGS
  STORE_EXIT=$?

  if [[ $STORE_EXIT -ne 0 ]]; then
    echo "[code-intel] Warning: AQE store had issues (exit $STORE_EXIT)" >&2
  fi
else
  echo "[code-intel] Skipping AQE memory store (--skip-store)" >&2
fi

# ---------------------------------------------------------------------------
# Post-index query if requested
# ---------------------------------------------------------------------------

if [[ -n "$QUERY" ]]; then
  echo "" >&2
  echo "[code-intel] Searching index for: $QUERY" >&2
  node -e "
    const data = require('$INDEX_FILE');
    const q = '$QUERY'.toLowerCase();
    const matches = data.entities
      .filter(e => e.name.toLowerCase().includes(q) || (e.file && e.file.toLowerCase().includes(q)))
      .slice(0, 20)
      .map(e => ({ name: e.name, type: e.type, file: e.file, line: e.line, project: e.project }));
    console.log(JSON.stringify({ query: '$QUERY', matches, total: matches.length }, null, 2));
  " 2>/dev/null
fi

# ---------------------------------------------------------------------------
# Final output (machine-readable for agents)
# ---------------------------------------------------------------------------

echo ""
echo "$INDEXER_RESULT"
