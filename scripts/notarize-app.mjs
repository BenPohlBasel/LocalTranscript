#!/usr/bin/env node
// Die gebaute .app notarisieren und das Ticket anheften — AUSSERHALB
// von `tauri build` (BACKLOG 9, Befund 2.4.1: Apples Warteschlange
// brauchte 22 min, Tauris interne Wartezeit lief ab und riss den Build
// mit leerer Meldung ab; die Einreichung lief bei Apple weiter). Hier:
// `notarytool submit --wait --timeout 2h`; ein Timeout kostet nie den
// Build. `tauri build` läuft dafür OHNE APPLE_*-Variablen (nur Signatur).
//
// Aufruf: node scripts/notarize-app.mjs [pfad.app]
// Zugang wie notarize-dmg.mjs: APPLE_KEYCHAIN_PROFILE oder
// APPLE_ID + APPLE_TEAM_ID + APPLE_PASSWORD.
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const app = process.argv[2]
  ?? path.join(ROOT, "frontend/src-tauri/target/release/bundle/macos/LocalTranscript.app");
if (!fs.existsSync(app)) {
  console.error(`Keine App unter ${app} — erst \`tauri build\`.`);
  process.exit(1);
}
const schon = spawnSync("/usr/bin/xcrun", ["stapler", "validate", app], { encoding: "utf8" });
if (schon.status === 0) {
  console.log(`Ticket hängt bereits an ${path.basename(app)} — nichts zu tun.`);
  process.exit(0);
}
const profil = process.env.APPLE_KEYCHAIN_PROFILE;
const zugang = profil
  ? ["--keychain-profile", profil]
  : ["--apple-id", process.env.APPLE_ID ?? "",
     "--team-id", process.env.APPLE_TEAM_ID ?? "",
     "--password", process.env.APPLE_PASSWORD ?? ""];
if (!profil && zugang.some((v) => v === "")) {
  console.error("Zugang fehlt: APPLE_KEYCHAIN_PROFILE oder APPLE_ID + APPLE_TEAM_ID + APPLE_PASSWORD.");
  process.exit(1);
}
// Apple will ein Zip der .app (ditto hält Ressourcen-Forks und Symlinks)
const zip = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "lt-notar-")), "LocalTranscript.zip");
execFileSync("/usr/bin/ditto", ["-c", "-k", "--keepParent", app, zip], { stdio: "inherit" });
console.log(`Reiche ein: ${path.basename(app)} (${(fs.statSync(zip).size / 1e6).toFixed(0)} MB) — Timeout 2 h.`);
try {
  execFileSync("/usr/bin/xcrun",
    ["notarytool", "submit", zip, ...zugang, "--wait", "--timeout", "2h"],
    { stdio: "inherit" });
  execFileSync("/usr/bin/xcrun", ["stapler", "staple", app], { stdio: "inherit" });
} catch {
  console.error("\nNotarisierung der App fehlgeschlagen — xcrun notarytool log <id> …");
  process.exit(1);
} finally {
  fs.rmSync(path.dirname(zip), { recursive: true, force: true });
}
