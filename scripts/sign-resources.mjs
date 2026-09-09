#!/usr/bin/env node
// Alle mitgelieferten Programmteile mit der Developer ID signieren —
// BEVOR Tauri das .app-Bundle versiegelt.
//
// Warum es diesen Schritt braucht: Tauri signiert nur die Hülle und das
// Hauptprogramm. Im Bundle stecken aber 253 weitere Mach-O-Dateien
// (python3, whisper-cli, ffmpeg, 244 dylib/so aus torch/numpy/
// SpeechBrain), und die tragen nur die Ad-hoc-Signatur, die der Linker
// vergibt. Apples Notardienst verlangt für JEDE ausführbare Datei die
// Developer ID samt Hardened Runtime — ohne diesen Lauf wird das
// Paket abgelehnt, und zwar erst nach dem 1,9-GB-Upload.
//
// Warum VOR dem Build: Signaturen liegen IM Mach-O, sie überleben das
// Kopieren. Tauri kopiert resources/ ins Bundle und versiegelt zuletzt
// die Hülle — die Reihenfolge stimmt also von selbst.
//
// Aufruf: node scripts/sign-resources.mjs [--identity <hash>]
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RES = path.join(ROOT, "frontend/src-tauri/resources");
const ENTITLEMENTS = path.join(ROOT, "frontend/src-tauri/entitlements.plist");
const CONF = path.join(ROOT, "frontend/src-tauri/tauri.conf.json");

function identitaet() {
  const i = process.argv.indexOf("--identity");
  if (i >= 0 && process.argv[i + 1]) return process.argv[i + 1];
  const c = JSON.parse(fs.readFileSync(CONF, "utf8"));
  const id = c?.bundle?.macOS?.signingIdentity;
  if (!id || id === "-") {
    console.error(
      "Keine Developer-ID in tauri.conf.json (signingIdentity ist "
      + `${id === "-" ? '"-", also ad-hoc' : "leer"}). `
      + "Ohne sie ist Signieren sinnlos — abgebrochen.");
    process.exit(1);
  }
  return id;
}

/** Mach-O? Die Magic-Bytes sagen es, die Endung lügt (viele .so und
    Programme ohne Endung, dazu Textdateien mit .dylib im Namen). */
function istMachO(datei) {
  let fd;
  try {
    fd = fs.openSync(datei, "r");
    const b = Buffer.alloc(4);
    if (fs.readSync(fd, b, 0, 4, 0) < 4) return false;
    const m = b.readUInt32BE(0);
    return m === 0xcffaedfe || m === 0xcefaedfe   // 64/32 little endian
        || m === 0xfeedfacf || m === 0xfeedface   // big endian
        || m === 0xcafebabe;                      // universal
  } catch { return false; } finally { if (fd !== undefined) fs.closeSync(fd); }
}

function sammle(dir, aus = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isSymbolicLink()) continue;        // Ziel wird selbst erfasst
    if (e.isDirectory()) sammle(p, aus);
    else if (e.isFile() && istMachO(p)) aus.push(p);
  }
  return aus;
}

const id = identitaet();
if (!fs.existsSync(RES)) {
  console.error(`resources/ fehlt (${RES}) — erst bundle-resources.mjs`);
  process.exit(1);
}
const dateien = sammle(RES);
console.log(`${dateien.length} Mach-O-Dateien unter resources/`);
console.log(`Identität: ${id}`);

let ok = 0;
const fehler = [];
const t0 = Date.now();
for (const [i, f] of dateien.entries()) {
  try {
    execFileSync("/usr/bin/codesign", [
      "--force", "--sign", id,
      "--options", "runtime",
      "--timestamp",
      "--entitlements", ENTITLEMENTS,
      f,
    ], { stdio: ["ignore", "ignore", "pipe"] });
    ok += 1;
  } catch (e) {
    fehler.push(`${path.relative(RES, f)}: ${String(e.stderr ?? e).trim()}`);
  }
  if ((i + 1) % 25 === 0 || i + 1 === dateien.length) {
    const s = (Date.now() - t0) / 1000;
    process.stdout.write(
      `\r  ${i + 1}/${dateien.length} · ${s.toFixed(0)} s`);
  }
}
process.stdout.write("\n");
console.log(`signiert: ${ok}${fehler.length ? `, Fehler: ${fehler.length}` : ""}`);
for (const f of fehler.slice(0, 10)) console.error("  " + f);
if (fehler.length) process.exit(1);
console.log("Fertig — jetzt `npx tauri build` (versiegelt die Hülle).");
