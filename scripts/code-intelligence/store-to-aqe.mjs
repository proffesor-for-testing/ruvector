#!/usr/bin/env node
/**
 * AQE Memory Store — Code Intelligence Results
 *
 * Reads the indexer output (index.json) and stores structured data into
 * AQE memory namespaces so QE agents can query it via:
 *   - aqe memory search --query "..." --namespace code-intelligence
 *   - mcp__agentic-qe__memory_query({ pattern: "code-intel/*", namespace: "code-intelligence" })
 *
 * Storage layout:
 *   code-intelligence/summary        — top-level scan summary
 *   code-intelligence/projects/*     — per-project entity/edge data
 *   code-intelligence/hotspots       — complexity hotspots list
 *   code-intelligence/cross-deps     — cross-project dependency graph
 *   code-intelligence/entity-index   — compact name->location lookup
 *   code-intelligence/circular-deps  — detected circular dependencies
 *
 * Usage:
 *   node scripts/code-intelligence/store-to-aqe.mjs [--input <path>] [--dry-run]
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { execSync } from 'node:child_process';

const ROOT = process.env.PROJECT_ROOT || findGitRoot();
const DEFAULT_INPUT = join(ROOT, '.agentic-qe', 'data', 'code-intel', 'index.json');
const NAMESPACE = 'code-intelligence';

const args = parseArgs(process.argv.slice(2));

async function main() {
  const inputPath = args.input || DEFAULT_INPUT;

  if (!existsSync(inputPath)) {
    console.error(`[store-aqe] Index file not found: ${inputPath}`);
    console.error('[store-aqe] Run the indexer first: node scripts/code-intelligence/index.mjs');
    process.exit(1);
  }

  const data = JSON.parse(readFileSync(inputPath, 'utf-8'));
  log(`Loaded index: ${data.summary.totalFiles} files, ${data.summary.totalEntities} entities`);

  const stores = [];

  // 1. Summary
  stores.push({
    key: 'code-intel/summary',
    value: {
      timestamp: data.timestamp,
      mode: data.mode,
      scanDurationMs: data.scanDurationMs,
      totalFiles: data.summary.totalFiles,
      totalEntities: data.summary.totalEntities,
      totalEdges: data.summary.totalEdges,
      crossProjectEdges: data.summary.crossProjectEdges,
      languageBreakdown: data.summary.languageBreakdown,
      subProjectCount: data.monorepo.subProjectCount,
      scannedProjectCount: data.monorepo.scannedProjectCount,
      averageComplexity: data.metrics.averageComplexityPerFile,
      totalLines: data.metrics.totalLines,
    },
  });

  // 2. Per-project summaries (compact — agents can load full project files separately)
  for (const proj of data.projects) {
    stores.push({
      key: `code-intel/projects/${proj.name}`,
      value: {
        name: proj.name,
        path: proj.path,
        type: proj.type,
        language: proj.language,
        fileCount: proj.fileCount,
        entityCount: proj.entityCount,
        edgeCount: proj.edgeCount,
        complexity: proj.complexity,
      },
    });
  }

  // 3. Hotspots (top 30 complexity hotspots)
  stores.push({
    key: 'code-intel/hotspots',
    value: {
      timestamp: data.timestamp,
      count: data.hotspots.length,
      items: data.hotspots,
    },
  });

  // 4. Cross-project dependencies
  const crossEdges = data.edges.filter(e => e.type === 'depends');
  stores.push({
    key: 'code-intel/cross-deps',
    value: {
      timestamp: data.timestamp,
      edgeCount: crossEdges.length,
      edges: crossEdges,
    },
  });

  // 5. Entity index (compact lookup: type -> name -> {file, line, project})
  // Build from entities, grouping by type
  const entityIndex = {};
  const entityTypes = new Set(data.entities.map(e => e.type));
  for (const type of entityTypes) {
    const ofType = data.entities.filter(e => e.type === type);
    entityIndex[type] = {
      count: ofType.length,
      // Store only first 200 per type to keep memory entry manageable
      items: ofType.slice(0, 200).map(e => ({
        name: e.name,
        file: e.file,
        line: e.line,
        project: e.project,
      })),
    };
  }
  stores.push({
    key: 'code-intel/entity-index',
    value: {
      timestamp: data.timestamp,
      totalEntities: data.entities.length,
      types: entityIndex,
    },
  });

  // 6. Circular dependencies
  if (data.circularDeps && data.circularDeps.length > 0) {
    stores.push({
      key: 'code-intel/circular-deps',
      value: {
        timestamp: data.timestamp,
        count: data.circularDeps.length,
        cycles: data.circularDeps,
      },
    });
  }

  // 7. Projects ranked by complexity (for risk-based testing)
  stores.push({
    key: 'code-intel/complexity-ranking',
    value: {
      timestamp: data.timestamp,
      ranking: data.metrics.projectsByComplexity,
    },
  });

  // Store all
  log(`Storing ${stores.length} entries in namespace "${NAMESPACE}"`);

  let stored = 0;
  let failed = 0;

  for (const entry of stores) {
    if (args.dryRun) {
      log(`[dry-run] Would store: ${entry.key} (${JSON.stringify(entry.value).length} bytes)`);
      stored++;
      continue;
    }

    try {
      storeToAqeMemory(entry.key, entry.value);
      stored++;
    } catch (err) {
      log(`Failed to store ${entry.key}: ${err.message}`);
      failed++;
    }
  }

  log(`Stored ${stored} entries, ${failed} failures`);

  console.log(JSON.stringify({
    status: failed === 0 ? 'success' : 'partial',
    stored,
    failed,
    namespace: NAMESPACE,
    keys: stores.map(s => s.key),
  }));
}

// ---------------------------------------------------------------------------
// AQE memory integration
// ---------------------------------------------------------------------------

function storeToAqeMemory(key, value) {
  // Write directly to the AQE data directory as JSON files (fast path)
  const memoryDir = join(ROOT, '.agentic-qe', 'data', 'code-intel', 'memory');
  mkdirSync(memoryDir, { recursive: true });
  const safeName = key.replace(/\//g, '__');
  writeFileSync(join(memoryDir, `${safeName}.json`), JSON.stringify({
    key,
    namespace: NAMESPACE,
    value,
    storedAt: new Date().toISOString(),
    tags: ['code-intelligence', 'index'],
  }, null, 2));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function escapeShell(str) {
  return str.replace(/'/g, "'\\''");
}

function findGitRoot() {
  try {
    return execSync('git rev-parse --show-toplevel', { encoding: 'utf-8' }).trim();
  } catch {
    return process.cwd();
  }
}

function parseArgs(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--input' && argv[i + 1]) result.input = argv[++i];
    else if (argv[i] === '--dry-run') result.dryRun = true;
  }
  return result;
}

function log(msg) {
  process.stderr.write(`[store-aqe] ${msg}\n`);
}

main().catch(err => {
  console.error(`[store-aqe] Fatal: ${err.message}`);
  process.exit(1);
});
