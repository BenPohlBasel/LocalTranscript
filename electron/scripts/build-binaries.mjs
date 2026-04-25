#!/usr/bin/env node
/**
 * Bundle the native binaries (whisper-cli, ffmpeg) and their dylibs into
 * electron/resources/bin and electron/resources/lib so the packaged app
 * works on Macs without Homebrew.
 *
 *   node scripts/build-binaries.mjs
 *
 * Inputs:
 *   - whisper-cli + libwhisper/libggml: copied from /opt/homebrew/Cellar/whisper-cpp/<latest>/libexec/
 *     (the libexec layout uses @loader_path/../lib so it stays standalone)
 *   - ffmpeg: from the ffmpeg-static npm package (statically linked, redistributable)
 */

import { mkdir, rm, cp, readdir, stat, copyFile, chmod } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const require = createRequire(import.meta.url);

const ELECTRON_DIR = path.resolve(__dirname, '..');
const RES_DIR = path.join(ELECTRON_DIR, 'resources');
const BIN_DIR = path.join(RES_DIR, 'bin');
const LIB_DIR = path.join(RES_DIR, 'lib');

const HOMEBREW_WHISPER = '/opt/homebrew/Cellar/whisper-cpp';

async function pickLatestVersion(dir) {
  const entries = await readdir(dir);
  // Pick highest version directory (lexical sort works for n.n.n)
  return entries.filter((e) => /^\d/.test(e)).sort().pop();
}

async function copyWhisperCpp() {
  if (!existsSync(HOMEBREW_WHISPER)) {
    throw new Error(
      `whisper-cpp not found at ${HOMEBREW_WHISPER}. Install it: brew install whisper-cpp`
    );
  }
  const version = await pickLatestVersion(HOMEBREW_WHISPER);
  if (!version) throw new Error('No whisper-cpp version directory found');
  const libexec = path.join(HOMEBREW_WHISPER, version, 'libexec');
  console.log(`whisper-cpp ${version} from ${libexec}`);

  await mkdir(BIN_DIR, { recursive: true });
  await mkdir(LIB_DIR, { recursive: true });

  // Copy the binary (dereference if it's a symlink)
  const whisperDest = path.join(BIN_DIR, 'whisper-cli');
  if (existsSync(whisperDest)) await rm(whisperDest, { force: true });
  await copyFile(path.join(libexec, 'bin', 'whisper-cli'), whisperDest);
  await chmod(whisperDest, 0o755);

  // Copy dylibs (skip non-dylib files like cmake/, pkgconfig/)
  const libSrc = path.join(libexec, 'lib');
  const libEntries = await readdir(libSrc);
  for (const entry of libEntries) {
    if (!entry.endsWith('.dylib')) continue;
    const srcPath = path.join(libSrc, entry);
    const destPath = path.join(LIB_DIR, entry);
    // Use cp so symlinks (libwhisper.dylib -> libwhisper.1.dylib) are preserved.
    await cp(srcPath, destPath, { dereference: false, verbatimSymlinks: true });
  }

  console.log(`+ whisper-cli + ${libEntries.filter(e => e.endsWith('.dylib')).length} dylibs`);
}

async function copyFfmpeg() {
  // The ffmpeg-static package exposes the binary path via its main export.
  let ffmpegPath;
  try {
    ffmpegPath = require('ffmpeg-static');
  } catch (e) {
    throw new Error(
      'ffmpeg-static not installed. Run `npm install` in the electron/ directory first.'
    );
  }
  if (!ffmpegPath || !existsSync(ffmpegPath)) {
    throw new Error(`ffmpeg-static binary missing at ${ffmpegPath}`);
  }
  console.log(`ffmpeg from ${ffmpegPath}`);

  await mkdir(BIN_DIR, { recursive: true });
  const dest = path.join(BIN_DIR, 'ffmpeg');
  if (existsSync(dest)) await rm(dest, { force: true });
  await copyFile(ffmpegPath, dest);
  await chmod(dest, 0o755);

  const s = await stat(dest);
  console.log(`+ ffmpeg (${(s.size / 1e6).toFixed(1)} MB)`);
}

async function main() {
  console.log('Building native binaries -> resources/{bin,lib}');
  await copyWhisperCpp();
  await copyFfmpeg();
  console.log('\nDone.');
  console.log(`  bin: ${BIN_DIR}`);
  console.log(`  lib: ${LIB_DIR}`);
}

main().catch((err) => {
  console.error('\nbuild-binaries failed:', err.message || err);
  process.exit(1);
});
