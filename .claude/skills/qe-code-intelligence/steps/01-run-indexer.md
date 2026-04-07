# Step 1: Run the Code Intelligence Indexer

## Invoking the Indexer

### Incremental Scan (default — fast, only changed files)
```bash
bash scripts/code-intelligence/run.sh --incremental
```

### Full Monorepo Scan
```bash
bash scripts/code-intelligence/run.sh --full --verbose
```

### Scoped Scan
```bash
bash scripts/code-intelligence/run.sh --scope crates --lang rs
bash scripts/code-intelligence/run.sh --scope npm/packages --lang ts,js
bash scripts/code-intelligence/run.sh --scope crates/ruvector-core
```

### Search Existing Index
```bash
bash scripts/code-intelligence/run.sh --query "HnswIndex"
```

## Output Files

| File | Purpose |
|------|---------|
| `.agentic-qe/data/code-intel/index.json` | Full index |
| `.agentic-qe/data/code-intel/entity-lookup.json` | Name->location map |
| `.agentic-qe/data/code-intel/projects/<name>.json` | Per-project detail |

## AQE Memory Keys (after store)

| Key | Content |
|-----|---------|
| `code-intel/summary` | Scan summary and language breakdown |
| `code-intel/projects/<name>` | Per-project stats |
| `code-intel/hotspots` | Top 30 complexity hotspots |
| `code-intel/cross-deps` | Cross-project dependency edges |
| `code-intel/entity-index` | Entity type->name->location |
| `code-intel/complexity-ranking` | Projects ranked by complexity |
