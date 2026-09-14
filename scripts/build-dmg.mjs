#!/usr/bin/env node
// DMG aus der (gestapelten) .app bauen und signieren — statt Tauris
// DMG-Schritt, der an dessen Notarisierung hängt (BACKLOG 9). Inhalt
// wie bei Tauri: die App und eine Verknüpfung auf /Applications.
//
// Aufruf: node scripts/build-dmg.mjs [pfad.app]  → bundle/dmg/TurnScript_<version>_aarch64.dmg
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const app = process.argv[2]
  ?? path.join(ROOT, "frontend/src-tauri/target/release/bundle/macos/TurnScript.app");
const conf = JSON.parse(fs.readFileSync(path.join(ROOT, "frontend/src-tauri/tauri.conf.json"), "utf8"));
const identity = conf.bundle?.macOS?.signingIdentity;
const version = conf.version;
const dmgDir = path.join(ROOT, "frontend/src-tauri/target/release/bundle/dmg");
const dmg = path.join(dmgDir, `TurnScript_${version}_aarch64.dmg`);
if (!fs.existsSync(app) || !identity) { console.error("App oder Signatur-Identität fehlt."); process.exit(1); }
fs.mkdirSync(dmgDir, { recursive: true });
const root = fs.mkdtempSync(path.join(os.tmpdir(), "lt-dmg-"));
execFileSync("/bin/cp", ["-R", app, root]);
fs.symlinkSync("/Applications", path.join(root, "Applications"));
fs.rmSync(dmg, { force: true });
execFileSync("/usr/bin/hdiutil", ["create", "-quiet", "-volname", "TurnScript",
  "-srcfolder", root, "-ov", "-format", "UDZO", "-imagekey", "zlib-level=9", dmg], { stdio: "inherit" });
execFileSync("/usr/bin/codesign", ["--force", "--sign", identity, "--timestamp", dmg], { stdio: "inherit" });
fs.rmSync(root, { recursive: true, force: true });
console.log(`DMG: ${dmg} (${(fs.statSync(dmg).size / 1e6).toFixed(0)} MB)`);
