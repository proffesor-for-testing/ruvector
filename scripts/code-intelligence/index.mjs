#!/usr/bin/env node
/**
 * Code Intelligence Incremental Indexer
 *
 * Scans the monorepo, extracts entities (files, functions, classes, modules),
 * maps relationships (imports, exports, dependencies), computes complexity
 * metrics, and writes structured JSON for AQE agent consumption.
 *
 * Supports incremental mode: only re-indexes files changed since last scan.
 *
 * Usage:
 *   node scripts/code-intelligence/index.mjs [options]
 *
 * Options:
 *   --scope <path>       Scope to a sub-project (e.g., crates/ruvector-core)
 *   --incremental        Only index files changed since last scan
 *   --since <iso-date>   Only index files changed since this date
 *   --output <path>      Output JSON path (default: .agentic-qe/data/code-intel/)
 *   --lang <lang,...>    Filter by language (ts,js,rs,py,go,java)
 *   --max-files <n>      Limit files per sub-project (default: 5000)
 *   --verbose            Verbose logging
 */

import { execSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { join, relative, extname, basename, dirname } from 'node:path';
import { createHash } from 'node:crypto';

// ---------------------------------------------------------------------------
// Config & CLI parsing
// ---------------------------------------------------------------------------

const ROOT = process.env.PROJECT_ROOT || findGitRoot();
const AQE_DIR = join(ROOT, '.agentic-qe');
const DEFAULT_OUTPUT = join(AQE_DIR, 'data', 'code-intel');
const STATE_FILE = join(DEFAULT_OUTPUT, '.index-state.json');

const LANG_EXTENSIONS = {
  ts: ['.ts', '.tsx'],
  js: ['.js', '.jsx', '.mjs', '.cjs'],
  rs: ['.rs'],
  py: ['.py'],
  go: ['.go'],
  java: ['.java'],
};

const IGNORE_DIRS = new Set([
  'node_modules', 'dist', 'build', 'target', '.git', '.next',
  'coverage', '.turbo', '.cache', '__pycache__', '.agentic-qe',
  'pkg', 'wasm-pack', '.wasm', 'vendor',
]);

const args = parseArgs(process.argv.slice(2));

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const startTime = Date.now();
  const outputDir = args.output || DEFAULT_OUTPUT;
  mkdirSync(outputDir, { recursive: true });

  const langs = args.lang
    ? args.lang.split(',').map(l => l.trim())
    : Object.keys(LANG_EXTENSIONS);

  const extensions = new Set(langs.flatMap(l => LANG_EXTENSIONS[l] || []));

  // Determine what to scan
  const subProjects = discoverSubProjects(ROOT, args.scope);
  log(`Discovered ${subProjects.length} sub-projects`);

  // Load previous state for incremental
  const prevState = loadState();
  const sinceDate = args.incremental
    ? (args.since || prevState.lastScanDate || null)
    : null;

  if (sinceDate) {
    log(`Incremental mode: only files changed since ${sinceDate}`);
  }

  const changedFiles = sinceDate ? getChangedFilesSince(sinceDate) : null;

  // Scan each sub-project
  const results = [];
  let totalFiles = 0;
  let totalEntities = 0;
  let totalEdges = 0;

  for (const project of subProjects) {
    const projectResult = scanProject(project, extensions, changedFiles, args.maxFiles || 5000);
    if (projectResult.files.length === 0) continue;

    results.push(projectResult);
    totalFiles += projectResult.files.length;
    totalEntities += projectResult.entities.length;
    totalEdges += projectResult.edges.length;

    if (args.verbose) {
      log(`  ${project.name}: ${projectResult.files.length} files, ` +
          `${projectResult.entities.length} entities, ${projectResult.edges.length} edges`);
    }
  }

  // Incremental merge: load previous index and patch in changed projects
  if (sinceDate && results.length === 0) {
    // Nothing changed — preserve existing index, just update timestamp
    const prevIndex = loadPreviousIndex(outputDir);
    if (prevIndex) {
      prevIndex.timestamp = new Date().toISOString();
      prevIndex.mode = 'incremental-noop';
      prevIndex.scanDurationMs = Date.now() - startTime;
      prevIndex.sinceDateUsed = sinceDate;
      writeFileSync(join(outputDir, 'index.json'), JSON.stringify(prevIndex, null, 2));
      saveState({
        lastScanDate: new Date().toISOString(),
        lastScanMode: 'incremental-noop',
        fileHashes: prevState.fileHashes || {},
        projectCount: prevIndex.monorepo.scannedProjectCount,
      });
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      log(`Done in ${elapsed}s: no changes detected, index preserved ` +
          `(${prevIndex.summary.totalFiles} files, ${prevIndex.summary.totalEntities} entities)`);
      console.log(JSON.stringify({
        status: 'success',
        outputPath: join(outputDir, 'index.json'),
        summary: prevIndex.summary,
        scanDurationMs: Date.now() - startTime,
      }));
      return;
    }
  }

  // If incremental with changes, merge with previous index
  if (sinceDate && results.length > 0) {
    const prevIndex = loadPreviousIndex(outputDir);
    if (prevIndex) {
      const changedProjectNames = new Set(results.map(r => r.project.name));
      // Keep previous results for unchanged projects
      const unchangedProjects = (prevIndex.projects || [])
        .filter(p => !changedProjectNames.has(p.name));
      // Merge entities/edges: remove old entries for changed projects, add new
      const unchangedEntities = (prevIndex.entities || [])
        .filter(e => !changedProjectNames.has(e.project));
      const unchangedEdges = (prevIndex.edges || [])
        .filter(e => {
          // Keep edges not from changed projects
          const proj = results.find(r => r.edges.some(re => re === e));
          return !proj;
        });
      // Merge into results
      for (const prevProj of unchangedProjects) {
        totalFiles += prevProj.fileCount || 0;
        totalEntities += prevProj.entityCount || 0;
        totalEdges += prevProj.edgeCount || 0;
      }
      // Prepend unchanged entities/edges
      results.unshift({
        project: { name: '__preserved__' },
        files: [],
        entities: unchangedEntities,
        edges: unchangedEdges.filter(e => e.type !== 'depends'),
        complexity: { average: 0, max: 0, maxFile: '', total: 0 },
        _preservedProjects: unchangedProjects,
      });
    }
  }

  // Build cross-project dependency edges
  const crossEdges = buildCrossProjectEdges(results);
  totalEdges += crossEdges.length;

  // Compute aggregate metrics
  const aggregateMetrics = computeAggregateMetrics(results);

  // Collect project summaries (exclude the __preserved__ placeholder)
  const projectSummaries = [];
  for (const r of results) {
    if (r._preservedProjects) {
      projectSummaries.push(...r._preservedProjects);
    } else {
      projectSummaries.push({
        name: r.project.name,
        path: r.project.relativePath,
        type: r.project.type,
        language: r.project.primaryLang,
        fileCount: r.files.length,
        entityCount: r.entities.length,
        edgeCount: r.edges.length,
        complexity: r.complexity,
      });
    }
  }

  // Assemble final output
  const output = {
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    mode: sinceDate ? 'incremental' : 'full',
    sinceDateUsed: sinceDate || null,
    scanDurationMs: Date.now() - startTime,
    monorepo: {
      root: ROOT,
      subProjectCount: subProjects.length,
      scannedProjectCount: projectSummaries.length,
    },
    summary: {
      totalFiles,
      totalEntities,
      totalEdges: totalEdges,
      crossProjectEdges: crossEdges.length,
      languageBreakdown: aggregateMetrics.languageBreakdown,
    },
    metrics: aggregateMetrics,
    projects: projectSummaries,
    entities: results.flatMap(r => r.entities),
    edges: [
      ...results.flatMap(r => r.edges),
      ...crossEdges,
    ],
    hotspots: identifyHotspots(results),
    circularDeps: detectCircularDeps(results),
  };

  // Write outputs
  const outPath = join(outputDir, 'index.json');
  writeFileSync(outPath, JSON.stringify(output, null, 2));

  // Write per-project indexes for targeted agent queries
  const projectDir = join(outputDir, 'projects');
  mkdirSync(projectDir, { recursive: true });
  for (const r of results) {
    const projFile = join(projectDir, `${r.project.name}.json`);
    writeFileSync(projFile, JSON.stringify({
      project: r.project,
      files: r.files,
      entities: r.entities,
      edges: r.edges,
      complexity: r.complexity,
    }, null, 2));
  }

  // Write compact entity lookup (name -> location)
  const entityLookup = {};
  for (const e of output.entities) {
    if (!entityLookup[e.type]) entityLookup[e.type] = {};
    entityLookup[e.type][e.name] = { file: e.file, line: e.line, project: e.project };
  }
  writeFileSync(join(outputDir, 'entity-lookup.json'), JSON.stringify(entityLookup, null, 2));

  // Save state for next incremental run
  saveState({
    lastScanDate: new Date().toISOString(),
    lastScanMode: sinceDate ? 'incremental' : 'full',
    fileHashes: buildFileHashMap(results),
    projectCount: results.length,
  });

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  log(`Done in ${elapsed}s: ${totalFiles} files, ${totalEntities} entities, ` +
      `${totalEdges} edges across ${results.length} projects`);
  log(`Output: ${outPath}`);

  // Print summary for agent consumption
  console.log(JSON.stringify({
    status: 'success',
    outputPath: outPath,
    summary: output.summary,
    scanDurationMs: output.scanDurationMs,
  }));
}

// ---------------------------------------------------------------------------
// Sub-project discovery
// ---------------------------------------------------------------------------

function discoverSubProjects(root, scope) {
  const projects = [];

  const scanDirs = [
    { dir: 'crates', type: 'rust-crate', lang: 'rs' },
    { dir: 'npm/packages', type: 'npm-package', lang: 'ts' },
    { dir: 'ui', type: 'ui-app', lang: 'ts' },
    { dir: 'examples', type: 'example', lang: 'ts' },
  ];

  for (const { dir, type, lang } of scanDirs) {
    const absDir = join(root, dir);
    if (!existsSync(absDir)) continue;

    const entries = readdirSync(absDir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      if (IGNORE_DIRS.has(entry.name)) continue;

      const fullPath = join(absDir, entry.name);
      const relPath = relative(root, fullPath);

      if (scope && !relPath.startsWith(scope) && !scope.startsWith(relPath)) {
        continue;
      }

      // Detect primary language from marker files
      let primaryLang = lang;
      if (existsSync(join(fullPath, 'Cargo.toml'))) primaryLang = 'rs';
      else if (existsSync(join(fullPath, 'package.json'))) primaryLang = 'ts';
      else if (existsSync(join(fullPath, 'go.mod'))) primaryLang = 'go';
      else if (existsSync(join(fullPath, 'setup.py')) || existsSync(join(fullPath, 'pyproject.toml'))) primaryLang = 'py';

      projects.push({
        name: entry.name,
        fullPath,
        relativePath: relPath,
        type,
        primaryLang,
      });
    }
  }

  // Also index root-level src/ if it exists and no scope filter
  const rootSrc = join(root, 'src');
  if (existsSync(rootSrc) && (!scope || scope === 'src')) {
    projects.push({
      name: 'root-src',
      fullPath: rootSrc,
      relativePath: 'src',
      type: 'root',
      primaryLang: 'ts',
    });
  }

  return projects;
}

// ---------------------------------------------------------------------------
// Project scanning
// ---------------------------------------------------------------------------

function scanProject(project, extensions, changedFiles, maxFiles) {
  const files = [];
  const entities = [];
  const edges = [];
  let complexitySum = 0;
  let maxComplexity = 0;
  let maxComplexityFile = '';

  // Collect source files
  const sourceFiles = collectFiles(project.fullPath, extensions, maxFiles);

  for (const filePath of sourceFiles) {
    const relPath = relative(ROOT, filePath);

    // Incremental: skip unchanged files
    if (changedFiles !== null && !changedFiles.has(relPath)) continue;

    const ext = extname(filePath);
    const content = safeReadFile(filePath);
    if (!content) continue;

    const lines = content.split('\n');
    const fileEntity = {
      id: hashId(relPath),
      name: basename(filePath),
      type: 'file',
      file: relPath,
      line: 1,
      project: project.name,
      language: extToLang(ext),
      lines: lines.length,
      complexity: 0,
    };

    files.push({ path: relPath, lines: lines.length, language: extToLang(ext) });

    // Extract entities based on language
    const extracted = extractEntities(content, relPath, ext, project.name);
    entities.push(fileEntity, ...extracted.entities);
    edges.push(...extracted.edges);

    // Compute file-level complexity
    const cx = computeComplexity(content, ext);
    fileEntity.complexity = cx;
    complexitySum += cx;
    if (cx > maxComplexity) {
      maxComplexity = cx;
      maxComplexityFile = relPath;
    }
  }

  return {
    project,
    files,
    entities,
    edges,
    complexity: {
      average: files.length > 0 ? Math.round((complexitySum / files.length) * 100) / 100 : 0,
      max: maxComplexity,
      maxFile: maxComplexityFile,
      total: complexitySum,
    },
  };
}

// ---------------------------------------------------------------------------
// Entity extraction (language-aware)
// ---------------------------------------------------------------------------

function extractEntities(content, filePath, ext, projectName) {
  const entities = [];
  const edges = [];
  const lang = extToLang(ext);

  const lines = content.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineNum = i + 1;

    if (lang === 'ts' || lang === 'js') {
      extractTsJsEntities(line, lineNum, filePath, projectName, entities, edges);
    } else if (lang === 'rs') {
      extractRustEntities(line, lineNum, filePath, projectName, entities, edges);
    } else if (lang === 'py') {
      extractPythonEntities(line, lineNum, filePath, projectName, entities, edges);
    } else if (lang === 'go') {
      extractGoEntities(line, lineNum, filePath, projectName, entities, edges);
    } else if (lang === 'java') {
      extractJavaEntities(line, lineNum, filePath, projectName, entities, edges);
    }
  }

  // Extract import edges
  const imports = extractImports(content, lang, filePath);
  edges.push(...imports);

  return { entities, edges };
}

function extractTsJsEntities(line, lineNum, filePath, projectName, entities, edges) {
  // Classes
  let m = line.match(/(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'class', file: filePath, line: lineNum, project: projectName });
    if (m[2]) edges.push({ source: hashId(`${filePath}:${m[1]}`), target: m[2], type: 'extends', file: filePath });
    if (m[3]) {
      for (const iface of m[3].split(',').map(s => s.trim())) {
        edges.push({ source: hashId(`${filePath}:${m[1]}`), target: iface, type: 'implements', file: filePath });
      }
    }
    return;
  }

  // Interfaces
  m = line.match(/(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'interface', file: filePath, line: lineNum, project: projectName });
    return;
  }

  // Type aliases
  m = line.match(/(?:export\s+)?type\s+(\w+)\s*[=<]/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'type', file: filePath, line: lineNum, project: projectName });
    return;
  }

  // Exported functions
  m = line.match(/(?:export\s+)?(?:async\s+)?function\s+(\w+)/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'function', file: filePath, line: lineNum, project: projectName });
    return;
  }

  // Arrow function exports: export const foo = (...) =>
  m = line.match(/export\s+const\s+(\w+)\s*=\s*(?:async\s+)?\(/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'function', file: filePath, line: lineNum, project: projectName });
    return;
  }

  // Enums
  m = line.match(/(?:export\s+)?(?:const\s+)?enum\s+(\w+)/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'enum', file: filePath, line: lineNum, project: projectName });
    return;
  }
}

function extractRustEntities(line, lineNum, filePath, projectName, entities, edges) {
  let m;

  // Structs
  m = line.match(/(?:pub(?:\([\w:]+\))?\s+)?struct\s+(\w+)/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'struct', file: filePath, line: lineNum, project: projectName });
    return;
  }

  // Enums
  m = line.match(/(?:pub(?:\([\w:]+\))?\s+)?enum\s+(\w+)/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'enum', file: filePath, line: lineNum, project: projectName });
    return;
  }

  // Traits
  m = line.match(/(?:pub(?:\([\w:]+\))?\s+)?trait\s+(\w+)/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'trait', file: filePath, line: lineNum, project: projectName });
    return;
  }

  // Impl blocks
  m = line.match(/impl(?:<[^>]+>)?\s+(\w+)(?:\s+for\s+(\w+))?/);
  if (m) {
    if (m[2]) {
      edges.push({ source: m[2], target: m[1], type: 'implements', file: filePath });
    }
    return;
  }

  // Functions
  m = line.match(/(?:pub(?:\([\w:]+\))?\s+)?(?:async\s+)?fn\s+(\w+)/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'function', file: filePath, line: lineNum, project: projectName });
    return;
  }

  // Modules
  m = line.match(/(?:pub\s+)?mod\s+(\w+)/);
  if (m && !line.includes('//')) {
    entities.push({ id: hashId(`${filePath}:mod:${m[1]}`), name: m[1], type: 'module', file: filePath, line: lineNum, project: projectName });
    return;
  }
}

function extractPythonEntities(line, lineNum, filePath, projectName, entities, edges) {
  let m;

  m = line.match(/^class\s+(\w+)(?:\(([^)]+)\))?/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'class', file: filePath, line: lineNum, project: projectName });
    if (m[2]) {
      for (const parent of m[2].split(',').map(s => s.trim())) {
        if (parent && parent !== 'object') {
          edges.push({ source: hashId(`${filePath}:${m[1]}`), target: parent, type: 'extends', file: filePath });
        }
      }
    }
    return;
  }

  m = line.match(/^(?:async\s+)?def\s+(\w+)/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'function', file: filePath, line: lineNum, project: projectName });
    return;
  }
}

function extractGoEntities(line, lineNum, filePath, projectName, entities, edges) {
  let m;

  m = line.match(/^type\s+(\w+)\s+struct/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'struct', file: filePath, line: lineNum, project: projectName });
    return;
  }

  m = line.match(/^type\s+(\w+)\s+interface/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'interface', file: filePath, line: lineNum, project: projectName });
    return;
  }

  m = line.match(/^func\s+(?:\(\w+\s+\*?(\w+)\)\s+)?(\w+)/);
  if (m) {
    const name = m[2];
    entities.push({ id: hashId(`${filePath}:${name}`), name, type: 'function', file: filePath, line: lineNum, project: projectName });
    if (m[1]) {
      edges.push({ source: hashId(`${filePath}:${name}`), target: m[1], type: 'uses', file: filePath });
    }
    return;
  }
}

function extractJavaEntities(line, lineNum, filePath, projectName, entities, edges) {
  let m;

  m = line.match(/(?:public|private|protected)?\s*(?:abstract\s+)?(?:static\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'class', file: filePath, line: lineNum, project: projectName });
    if (m[2]) edges.push({ source: hashId(`${filePath}:${m[1]}`), target: m[2], type: 'extends', file: filePath });
    if (m[3]) {
      for (const iface of m[3].split(',').map(s => s.trim())) {
        edges.push({ source: hashId(`${filePath}:${m[1]}`), target: iface, type: 'implements', file: filePath });
      }
    }
    return;
  }

  m = line.match(/(?:public|private|protected)?\s*interface\s+(\w+)/);
  if (m) {
    entities.push({ id: hashId(`${filePath}:${m[1]}`), name: m[1], type: 'interface', file: filePath, line: lineNum, project: projectName });
    return;
  }
}

// ---------------------------------------------------------------------------
// Import extraction
// ---------------------------------------------------------------------------

function extractImports(content, lang, filePath) {
  const edges = [];

  if (lang === 'ts' || lang === 'js') {
    // import ... from '...'
    const re = /import\s+.*?from\s+['"]([^'"]+)['"]/g;
    let m;
    while ((m = re.exec(content)) !== null) {
      edges.push({ source: filePath, target: m[1], type: 'imports', file: filePath });
    }
    // require('...')
    const re2 = /require\s*\(\s*['"]([^'"]+)['"]\s*\)/g;
    while ((m = re2.exec(content)) !== null) {
      edges.push({ source: filePath, target: m[1], type: 'imports', file: filePath });
    }
  } else if (lang === 'rs') {
    const re = /use\s+([\w:]+)/g;
    let m;
    while ((m = re.exec(content)) !== null) {
      edges.push({ source: filePath, target: m[1], type: 'imports', file: filePath });
    }
    // extern crate
    const re2 = /extern\s+crate\s+(\w+)/g;
    while ((m = re2.exec(content)) !== null) {
      edges.push({ source: filePath, target: m[1], type: 'imports', file: filePath });
    }
  } else if (lang === 'py') {
    const re = /(?:from\s+([\w.]+)\s+)?import\s+([\w.,\s]+)/g;
    let m;
    while ((m = re.exec(content)) !== null) {
      const mod = m[1] || m[2].split(',')[0].trim();
      edges.push({ source: filePath, target: mod, type: 'imports', file: filePath });
    }
  } else if (lang === 'go') {
    const re = /import\s+(?:\(\s*([\s\S]*?)\s*\)|"([^"]+)")/g;
    let m;
    while ((m = re.exec(content)) !== null) {
      if (m[2]) {
        edges.push({ source: filePath, target: m[2], type: 'imports', file: filePath });
      } else if (m[1]) {
        for (const imp of m[1].match(/"([^"]+)"/g) || []) {
          edges.push({ source: filePath, target: imp.replace(/"/g, ''), type: 'imports', file: filePath });
        }
      }
    }
  } else if (lang === 'java') {
    const re = /import\s+([\w.]+);/g;
    let m;
    while ((m = re.exec(content)) !== null) {
      edges.push({ source: filePath, target: m[1], type: 'imports', file: filePath });
    }
  }

  return edges;
}

// ---------------------------------------------------------------------------
// Complexity computation (cyclomatic approximation)
// ---------------------------------------------------------------------------

function computeComplexity(content, ext) {
  // Cyclomatic complexity approximation: count decision points
  const lang = extToLang(ext);
  let complexity = 1; // base path

  const decisionPatterns = {
    ts: /\b(if|else\s+if|for|while|do|switch|case|catch|\?\?|&&|\|\||[\?](?=\s*[^:]))\b/g,
    js: /\b(if|else\s+if|for|while|do|switch|case|catch|\?\?|&&|\|\||[\?](?=\s*[^:]))\b/g,
    rs: /\b(if|else\s+if|for|while|loop|match|=>|&&|\|\|)\b/g,
    py: /\b(if|elif|for|while|except|and|or|with)\b/g,
    go: /\b(if|else\s+if|for|switch|case|select|&&|\|\|)\b/g,
    java: /\b(if|else\s+if|for|while|do|switch|case|catch|&&|\|\|)\b/g,
  };

  const pattern = decisionPatterns[lang];
  if (pattern) {
    const matches = content.match(pattern);
    if (matches) complexity += matches.length;
  }

  return complexity;
}

// ---------------------------------------------------------------------------
// Cross-project edges
// ---------------------------------------------------------------------------

function buildCrossProjectEdges(results) {
  const edges = [];
  const projectExports = new Map(); // name -> project

  // First pass: collect all exported entity names per project
  for (const r of results) {
    for (const e of r.entities) {
      if (e.type !== 'file') {
        projectExports.set(e.name, r.project.name);
      }
    }
  }

  // Second pass: look for import edges that reference entities in other projects
  for (const r of results) {
    for (const edge of r.edges) {
      if (edge.type === 'imports') {
        const target = edge.target;
        // Check if import target contains a known project name
        for (const [, projName] of projectExports) {
          if (projName !== r.project.name && target.includes(projName.replace(/-/g, '_'))) {
            edges.push({
              source: r.project.name,
              target: projName,
              type: 'depends',
              via: target,
            });
            break;
          }
        }
      }
    }
  }

  // Deduplicate
  const seen = new Set();
  return edges.filter(e => {
    const key = `${e.source}:${e.target}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

// ---------------------------------------------------------------------------
// Hotspot detection
// ---------------------------------------------------------------------------

function identifyHotspots(results, topN = 30) {
  const allFiles = [];

  for (const r of results) {
    for (const f of r.files) {
      // Score = lines * log(complexity+1) weighted by entity density
      const entityCount = r.entities.filter(e => e.file === f.path).length;
      const complexity = r.entities
        .filter(e => e.file === f.path && e.complexity)
        .reduce((sum, e) => sum + (e.complexity || 0), 0);

      const score = f.lines * Math.log2(complexity + 2) * (1 + entityCount * 0.1);
      allFiles.push({
        path: f.path,
        project: r.project.name,
        type: 'complexity',
        score: Math.round(score * 100) / 100,
        lines: f.lines,
        entityCount,
      });
    }
  }

  allFiles.sort((a, b) => b.score - a.score);
  return allFiles.slice(0, topN);
}

// ---------------------------------------------------------------------------
// Circular dependency detection
// ---------------------------------------------------------------------------

function detectCircularDeps(results) {
  // Build adjacency list from import edges
  const adj = new Map();
  for (const r of results) {
    for (const edge of r.edges) {
      if (edge.type === 'imports') {
        if (!adj.has(edge.source)) adj.set(edge.source, new Set());
        adj.get(edge.source).add(edge.target);
      }
    }
  }

  // DFS-based cycle detection (limited to short cycles to avoid blowup)
  const cycles = [];
  const visited = new Set();
  const stack = new Set();
  const path = [];

  function dfs(node, depth) {
    if (depth > 5) return; // limit traversal depth
    if (stack.has(node)) {
      const cycleStart = path.indexOf(node);
      if (cycleStart !== -1) {
        cycles.push(path.slice(cycleStart).concat(node));
      }
      return;
    }
    if (visited.has(node)) return;

    visited.add(node);
    stack.add(node);
    path.push(node);

    for (const neighbor of (adj.get(node) || [])) {
      if (cycles.length < 20) { // cap cycles reported
        dfs(neighbor, depth + 1);
      }
    }

    stack.delete(node);
    path.pop();
  }

  for (const node of adj.keys()) {
    if (!visited.has(node) && cycles.length < 20) {
      dfs(node, 0);
    }
  }

  return cycles;
}

// ---------------------------------------------------------------------------
// Aggregate metrics
// ---------------------------------------------------------------------------

function computeAggregateMetrics(results) {
  const langCounts = {};
  let totalLines = 0;
  let totalComplexity = 0;

  for (const r of results) {
    for (const f of r.files) {
      langCounts[f.language] = (langCounts[f.language] || 0) + 1;
      totalLines += f.lines;
    }
    totalComplexity += r.complexity.total;
  }

  const totalFiles = results.reduce((s, r) => s + r.files.length, 0);

  return {
    totalLines,
    averageComplexityPerFile: totalFiles > 0 ? Math.round((totalComplexity / totalFiles) * 100) / 100 : 0,
    languageBreakdown: langCounts,
    projectsByComplexity: results
      .map(r => ({ name: r.project.name, complexity: r.complexity.average }))
      .sort((a, b) => b.complexity - a.complexity)
      .slice(0, 20),
  };
}

// ---------------------------------------------------------------------------
// Git helpers
// ---------------------------------------------------------------------------

function getChangedFilesSince(sinceDate) {
  try {
    const cmd = `git -C "${ROOT}" log --since="${sinceDate}" --name-only --pretty=format: --diff-filter=ACMR`;
    const output = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
    return new Set(output.split('\n').map(l => l.trim()).filter(Boolean));
  } catch {
    return null; // fall back to full scan
  }
}

// ---------------------------------------------------------------------------
// File utilities
// ---------------------------------------------------------------------------

function collectFiles(dir, extensions, maxFiles) {
  const files = [];
  const queue = [dir];

  while (queue.length > 0 && files.length < maxFiles) {
    const current = queue.shift();
    let entries;
    try {
      entries = readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      if (files.length >= maxFiles) break;
      const full = join(current, entry.name);

      if (entry.isDirectory()) {
        if (!IGNORE_DIRS.has(entry.name) && !entry.name.startsWith('.')) {
          queue.push(full);
        }
      } else if (entry.isFile() && extensions.has(extname(entry.name))) {
        files.push(full);
      }
    }
  }

  return files;
}

function safeReadFile(filePath) {
  try {
    const stat = statSync(filePath);
    if (stat.size > 512 * 1024) return null; // skip files > 512KB
    return readFileSync(filePath, 'utf-8');
  } catch {
    return null;
  }
}

function findGitRoot() {
  try {
    return execSync('git rev-parse --show-toplevel', { encoding: 'utf-8' }).trim();
  } catch {
    return process.cwd();
  }
}

// ---------------------------------------------------------------------------
// State management (for incremental mode)
// ---------------------------------------------------------------------------

function loadPreviousIndex(outputDir) {
  const indexPath = join(outputDir, 'index.json');
  if (!existsSync(indexPath)) return null;
  try {
    return JSON.parse(readFileSync(indexPath, 'utf-8'));
  } catch {
    return null;
  }
}

function loadState() {
  if (!existsSync(STATE_FILE)) return {};
  try {
    return JSON.parse(readFileSync(STATE_FILE, 'utf-8'));
  } catch {
    return {};
  }
}

function saveState(state) {
  writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

function buildFileHashMap(results) {
  const map = {};
  for (const r of results) {
    for (const f of r.files) {
      map[f.path] = f.lines; // lightweight: use line count as proxy
    }
  }
  return map;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function hashId(input) {
  return createHash('md5').update(input).digest('hex').slice(0, 12);
}

function extToLang(ext) {
  for (const [lang, exts] of Object.entries(LANG_EXTENSIONS)) {
    if (exts.includes(ext)) return lang;
  }
  return 'unknown';
}

function parseArgs(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--scope' && argv[i + 1]) result.scope = argv[++i];
    else if (arg === '--output' && argv[i + 1]) result.output = argv[++i];
    else if (arg === '--lang' && argv[i + 1]) result.lang = argv[++i];
    else if (arg === '--since' && argv[i + 1]) result.since = argv[++i];
    else if (arg === '--max-files' && argv[i + 1]) result.maxFiles = parseInt(argv[++i], 10);
    else if (arg === '--incremental') result.incremental = true;
    else if (arg === '--verbose') result.verbose = true;
  }
  return result;
}

function log(msg) {
  process.stderr.write(`[code-intel] ${msg}\n`);
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

main().catch(err => {
  console.error(`[code-intel] Fatal: ${err.message}`);
  process.exit(1);
});
