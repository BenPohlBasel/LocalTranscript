#!/usr/bin/env node
// Das DMG notarisieren und das Ticket anheften.
//
// Warum ein eigener Schritt: Tauri notarisiert nur die .app. Das DMG
// signiert es zwar mit derselben Identität, reicht es aber nie bei
// Apple ein — Gatekeeper sagt beim Doppelklick auf das geladene Image
// dann „Unnotarized Developer ID" (Befund 2026-09-09, von Hand
// nachgereicht). Angeheftet gilt das Ticket auch OHNE Netz.
//
// Aufruf: node scripts/notarize-dmg.mjs [pfad.dmg]
// Ohne Pfad wird das neueste DMG im Bundle-Ordner genommen.
//
// Erwartet APPLE_ID, APPLE_PASSWORD (app-spezifisch), APPLE_TEAM_ID —
// oder ein hinterlegtes Profil in APPLE_KEYCHAIN_PROFILE (dann muss
// kein Passwort in der Umgebung stehen).
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DMG_DIR = path.join(ROOT,
  "frontend/src-tauri/target/release/bundle/dmg");

function neuestesDmg() {
  if (!fs.existsSync(DMG_DIR)) return null;
  const kandidaten = fs.readdirSync(DMG_DIR)
    .filter((n) => n.endsWith(".dmg"))
    .map((n) => path.join(DMG_DIR, n))
    .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  return kandidaten[0] ?? null;
}

const dmg = process.argv[2] ?? neuestesDmg();
if (!dmg || !fs.existsSync(dmg)) {
  console.error(`Kein DMG gefunden (${DMG_DIR}) — erst \`tauri build\`.`);
  process.exit(1);
}

// Schon angeheftet? Dann ist nichts zu tun — spart einen 1,8-GB-Upload.
const schon = spawnSync("/usr/bin/xcrun", ["stapler", "validate", dmg],
                        { encoding: "utf8" });
if (schon.status === 0) {
  console.log(`Ticket hängt bereits an ${path.basename(dmg)} — nichts zu tun.`);
  process.exit(0);
}

const profil = process.env.APPLE_KEYCHAIN_PROFILE;
const zugang = profil
  ? ["--keychain-profile", profil]
  : ["--apple-id", process.env.APPLE_ID ?? "",
     "--team-id", process.env.APPLE_TEAM_ID ?? "",
     "--password", process.env.APPLE_PASSWORD ?? ""];
if (!profil && zugang.some((v) => v === "")) {
  console.error(
    "Zugang fehlt: entweder APPLE_KEYCHAIN_PROFILE oder APPLE_ID + "
    + "APPLE_TEAM_ID + APPLE_PASSWORD setzen.\n"
    + "Profil anlegen: xcrun notarytool store-credentials \"localtranscript\" "
    + "--apple-id … --team-id … --password …");
  process.exit(1);
}

const mb = (fs.statSync(dmg).size / 1e6).toFixed(0);
console.log(`Reiche ein: ${path.basename(dmg)} (${mb} MB) — das dauert.`);
try {
  execFileSync("/usr/bin/xcrun",
    ["notarytool", "submit", dmg, ...zugang, "--wait"],
    { stdio: "inherit" });
  execFileSync("/usr/bin/xcrun", ["stapler", "staple", dmg],
    { stdio: "inherit" });
} catch {
  console.error("\nNotarisierung fehlgeschlagen. Den Grund zeigt:\n"
    + "  xcrun notarytool log <submission-id> …");
  process.exit(1);
}

// Gegenprobe: würde Gatekeeper das Image jetzt öffnen?
const urteil = spawnSync("/usr/sbin/spctl",
  ["-a", "-t", "open", "--context", "context:primary-signature", "-vv", dmg],
  { encoding: "utf8" });
console.log((urteil.stderr || urteil.stdout).trim());
