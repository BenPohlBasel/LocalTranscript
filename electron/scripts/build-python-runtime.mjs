#!/usr/bin/env node
/**
 * Build a relocatable Python runtime + venv for the macOS Electron bundle.
 *
 *   node scripts/build-python-runtime.mjs           # idempotent, skip if present
 *   node scripts/build-python-runtime.mjs --force   # wipe and rebuild
 *
 * Output layout (under electron/resources/):
 *   python-runtime/   -> python-build-standalone CPython tree
 *   venv/             -> venv created from the runtime, with backend deps installed
 */

import { mkdir, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ELECTRON_DIR = path.resolve(__dirname, '..');
const PROJECT_ROOT = path.resolve(ELECTRON_DIR, '..');
const RES_DIR = path.join(ELECTRON_DIR, 'resources');
const RUNTIME_DIR = path.join(RES_DIR, 'python-runtime');
const VENV_DIR = path.join(RES_DIR, 'venv');
const BACKEND_REQUIREMENTS = path.join(PROJECT_ROOT, 'backend', 'requirements.txt');

const PYTHON_VERSION = process.env.PYTHON_VERSION || '3.13';
const ARCH = process.env.PBS_ARCH || 'aarch64-apple-darwin';

const force = process.argv.includes('--force');

function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    console.log(`+ ${cmd} ${args.map(a => a.includes(' ') ? `"${a}"` : a).join(' ')}`);
    const child = spawn(cmd, args, { stdio: 'inherit', ...opts });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} exited with code ${code}`));
    });
  });
}

async function findLatestRelease() {
  console.log('Querying GitHub for latest python-build-standalone release...');
  const headers = { 'User-Agent': 'whisper-web-build' };
  if (process.env.GITHUB_TOKEN) {
    headers['Authorization'] = `Bearer ${process.env.GITHUB_TOKEN}`;
  }
  const res = await fetch(
    'https://api.github.com/repos/astral-sh/python-build-standalone/releases?per_page=20',
    { headers }
  );
  if (!res.ok) {
    throw new Error(`GitHub API error: ${res.status} ${res.statusText}`);
  }
  const releases = await res.json();
  // Standard ends with `${ARCH}-install_only.tar.gz`. Free-threaded variant
  // is `${ARCH}-freethreaded-install_only.tar.gz`. We want the standard one.
  const wantSuffix = `${ARCH}-install_only.tar.gz`;
  for (const release of releases) {
    for (const asset of release.assets || []) {
      const n = asset.name;
      if (
        n.startsWith(`cpython-${PYTHON_VERSION}.`) &&
        n.endsWith(wantSuffix) &&
        !n.includes('freethreaded')
      ) {
        return { url: asset.browser_download_url, name: n, release: release.tag_name };
      }
    }
  }
  throw new Error(`No PBS asset found for cpython-${PYTHON_VERSION}.x ${ARCH}`);
}

async function downloadFile(url, dest) {
  console.log(`Downloading ${path.basename(dest)} ...`);
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await mkdir(path.dirname(dest), { recursive: true });
  await writeFile(dest, buf);
  console.log(`  -> ${(buf.length / 1024 / 1024).toFixed(1)} MB`);
}

async function ensurePythonRuntime() {
  if (existsSync(RUNTIME_DIR) && !force) {
    console.log(`python-runtime already present at ${RUNTIME_DIR}, skipping download`);
    return;
  }
  if (existsSync(RUNTIME_DIR)) {
    console.log('Removing existing python-runtime...');
    await rm(RUNTIME_DIR, { recursive: true, force: true });
  }

  const { url, name, release } = await findLatestRelease();
  console.log(`Selected: ${name} (release ${release})`);

  const tmpDir = path.join(RES_DIR, '.tmp');
  await mkdir(tmpDir, { recursive: true });
  const tarball = path.join(tmpDir, name);
  await downloadFile(url, tarball);

  await mkdir(RUNTIME_DIR, { recursive: true });
  // PBS tarballs extract to a top-level "python/" directory; strip it.
  await run('tar', ['-xzf', tarball, '-C', RUNTIME_DIR, '--strip-components=1']);
  await rm(tmpDir, { recursive: true, force: true });

  console.log('Python runtime extracted.');
}

function runtimePython() {
  return path.join(RUNTIME_DIR, 'bin', 'python3');
}

async function ensureVenv() {
  if (existsSync(VENV_DIR) && !force) {
    console.log(`venv already present at ${VENV_DIR}, skipping creation`);
    return;
  }
  if (existsSync(VENV_DIR)) {
    console.log('Removing existing venv...');
    await rm(VENV_DIR, { recursive: true, force: true });
  }
  // PBS on macOS uses `@rpath/libpython3.13.dylib`. With `--copies` venv
  // strips that rpath and the venv binary fails to load libpython. With
  // symlinks the rpath is preserved and resolves back into the runtime.
  // Both runtime and venv ship in the same Resources directory in the
  // bundle, so symlinks remain valid.
  await run(runtimePython(), ['-m', 'venv', VENV_DIR]);
}

async function installDeps() {
  const pip = path.join(VENV_DIR, 'bin', 'pip');
  await run(pip, ['install', '--upgrade', 'pip']);
  await run(pip, ['install', '-r', BACKEND_REQUIREMENTS]);
}

async function smokeTest() {
  const py = path.join(VENV_DIR, 'bin', 'python');
  await run(py, [
    '-c',
    'import fastapi, uvicorn, torch, torchaudio, speechbrain, silero_vad, sklearn; print("imports ok"); print("torch", torch.__version__); print("speechbrain", speechbrain.__version__)',
  ]);
}

async function main() {
  console.log(`Target: Python ${PYTHON_VERSION} (${ARCH})`);
  console.log(`Output: ${RES_DIR}`);
  console.log('');
  await ensurePythonRuntime();
  await ensureVenv();
  await installDeps();
  await smokeTest();
  console.log('');
  console.log('Done.');
  console.log(`  Runtime: ${RUNTIME_DIR}`);
  console.log(`  Venv:    ${VENV_DIR}`);
}

main().catch((err) => {
  console.error('\nBuild failed:', err.message || err);
  process.exit(1);
});
