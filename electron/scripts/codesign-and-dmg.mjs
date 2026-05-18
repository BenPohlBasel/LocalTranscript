#!/usr/bin/env node
/**
 * Post-process the electron-builder output:
 *   1. Ad-hoc codesign the .app (`codesign --force --deep --sign -`)
 *   2. Repackage into a fresh DMG containing the now-signed bundle plus an
 *      /Applications drag-target symlink.
 *
 * Why: electron-builder's `identity: "-"` is interpreted as a keychain
 * identity NAME, not as the ad-hoc placeholder Apple's `codesign` uses.
 * Setting it to `null` skips signing entirely, which fails on Apple Silicon
 * with the "is damaged" Gatekeeper message. Doing the ad-hoc sign manually
 * after the build avoids both issues without needing an Apple Developer ID.
 *
 *   node scripts/codesign-and-dmg.mjs
 */

import { spawn } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { rm, mkdir, cp, symlink, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ELECTRON_DIR = path.resolve(__dirname, '..');
const DIST_DIR = path.join(ELECTRON_DIR, 'dist');
const APP_PATH = path.join(DIST_DIR, 'mac-arm64', 'LocalTranscript.app');

function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    console.log(`+ ${cmd} ${args.map(a => a.includes(' ') ? `"${a}"` : a).join(' ')}`);
    const child = spawn(cmd, args, { stdio: 'inherit', ...opts });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} exited ${code}`));
    });
  });
}

async function main() {
  if (!existsSync(APP_PATH)) {
    throw new Error(`Bundle not found at ${APP_PATH}. Run \`npm run dist\` first.`);
  }

  const pkg = JSON.parse(await readFile(path.join(ELECTRON_DIR, 'package.json'), 'utf-8'));
  const version = pkg.version;
  const dmgName = `LocalTranscript-${version}-arm64.dmg`;
  const dmgPath = path.join(DIST_DIR, dmgName);

  console.log(`\n=== Ad-hoc codesign ${APP_PATH} ===`);
  await run('codesign', ['--force', '--deep', '--sign', '-', APP_PATH]);

  console.log('\n=== Verify signature ===');
  await run('codesign', ['--verify', '--verbose=2', APP_PATH]);

  console.log(`\n=== Repackage DMG -> ${dmgName} ===`);
  if (existsSync(dmgPath)) {
    await rm(dmgPath, { force: true });
  }
  const blockmap = dmgPath + '.blockmap';
  if (existsSync(blockmap)) {
    await rm(blockmap, { force: true });
  }

  const stage = path.join(DIST_DIR, '.dmg-stage');
  await rm(stage, { recursive: true, force: true });
  await mkdir(stage, { recursive: true });
  await cp(APP_PATH, path.join(stage, 'LocalTranscript.app'), { recursive: true });
  await symlink('/Applications', path.join(stage, 'Applications'));

  await run('hdiutil', [
    'create',
    '-volname', `LocalTranscript ${version}`,
    '-srcfolder', stage,
    '-ov',
    '-format', 'UDZO',
    dmgPath,
  ]);

  await rm(stage, { recursive: true, force: true });

  const size = statSync(dmgPath).size;
  console.log(`\nDone. ${dmgName} (${(size / 1e9).toFixed(2)} GB)`);
}

main().catch((err) => {
  console.error('\ncodesign-and-dmg failed:', err.message || err);
  process.exit(1);
});
