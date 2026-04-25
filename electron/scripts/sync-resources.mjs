#!/usr/bin/env node
/**
 * Copy backend code, binaries, models and .env into electron/resources/.
 *
 * Run before each Electron build. Idempotent — overwrites mirror copies.
 *
 * Layout produced:
 *   electron/resources/
 *     backend/         (Python source)
 *     bin/             (whisper-cli, ffmpeg)
 *     models/          (ggml-*.bin)
 *     .env             (HF_TOKEN; copied if present in project root)
 *
 * Note: python-runtime/ and venv/ are produced by build-python-runtime.mjs
 * and live next to these.
 */

import { mkdir, rm, cp, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ELECTRON_DIR = path.resolve(__dirname, '..');
const PROJECT_ROOT = path.resolve(ELECTRON_DIR, '..');
const RES_DIR = path.join(ELECTRON_DIR, 'resources');

// Note: bin/ and lib/ are produced by build-binaries.mjs (whisper-cli + dylibs
// from Homebrew, ffmpeg from ffmpeg-static), not copied from the project.
const ITEMS = [
  { src: 'backend', dest: 'backend', exclude: ['__pycache__'] },
  { src: 'frontend', dest: 'frontend' },
  { src: 'models', dest: 'models' },
  { src: '.env', dest: '.env', optional: true },
];

async function fileSize(p) {
  try {
    const s = await stat(p);
    if (s.isDirectory()) return null;
    return s.size;
  } catch {
    return null;
  }
}

async function dirSize(dir) {
  // Cheap recursive size; only used for reporting.
  const { readdir } = await import('node:fs/promises');
  let total = 0;
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return 0;
  }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) total += await dirSize(p);
    else {
      try { total += (await stat(p)).size; } catch {}
    }
  }
  return total;
}

function fmt(bytes) {
  if (bytes == null) return '?';
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

async function syncOne(item) {
  const src = path.join(PROJECT_ROOT, item.src);
  const dest = path.join(RES_DIR, item.dest);
  if (!existsSync(src)) {
    if (item.optional) {
      console.log(`- skipped ${item.src} (not present)`);
      return;
    }
    throw new Error(`Required source missing: ${src}`);
  }

  if (existsSync(dest)) {
    await rm(dest, { recursive: true, force: true });
  }

  const filter = item.exclude
    ? (s) => !item.exclude.some((ex) => s.includes(`${path.sep}${ex}${path.sep}`) || s.endsWith(`${path.sep}${ex}`))
    : undefined;

  // dereference: follow symlinks (models/ is symlinked to ~/whisper-models in dev)
  await cp(src, dest, { recursive: true, dereference: true, filter });

  const size = (await fileSize(dest)) ?? (await dirSize(dest));
  console.log(`+ ${item.src} -> resources/${item.dest}  (${fmt(size)})`);
}

async function main() {
  await mkdir(RES_DIR, { recursive: true });
  for (const item of ITEMS) {
    await syncOne(item);
  }

  // Marker file so the bundled backend can detect bundled mode at runtime.
  await (await import('node:fs/promises')).writeFile(
    path.join(RES_DIR, 'BUNDLED'),
    `built ${new Date().toISOString()}\n`,
    'utf-8'
  );

  console.log('\nResources ready.');
}

main().catch((err) => {
  console.error('\nSync failed:', err.message || err);
  process.exit(1);
});
