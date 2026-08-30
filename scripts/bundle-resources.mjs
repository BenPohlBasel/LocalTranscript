#!/usr/bin/env node
// Bundle-Resources für den Tauri-Build zusammenstellen:
// python-runtime + whisper-cli/dylibs + ffmpeg + Modelle werden aus dem
// alten LocalTranscript-v1-Checkout ÜBERNOMMEN, wenn er daneben liegt
// (whisper-web/electron/resources — dort hat build-python-runtime.mjs
// sie einst gebaut); sonst bricht das Skript mit Anleitung ab.
// Das venv wird IMMER FRISCH gebaut (v2-Backend + enrich-core).
//
// Aufruf: node scripts/bundle-resources.mjs [--force-venv]
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RES = path.join(ROOT, "frontend/src-tauri/resources");
const ALT = path.resolve(ROOT, "../whisper-web/electron/resources");
const ENRICH_CORE = path.resolve(ROOT, "../enrich/packages/enrich-core");
const forceVenv = process.argv.includes("--force-venv");

function da(p) { try { return fs.statSync(p).isDirectory(); } catch { return false; } }
function leer(p) { try { return fs.readdirSync(p).length === 0; } catch { return true; } }
function kopiere(von, nach) {
  console.log(`kopiere ${von} → ${nach}`);
  fs.rmSync(nach, { recursive: true, force: true });
  fs.cpSync(von, nach, { recursive: true, verbatimSymlinks: true });
}

// 1. Runtime + Binaries + Modelle
for (const teil of ["python-runtime", "bin", "lib", "models"]) {
  const ziel = path.join(RES, teil);
  if (da(ziel) && !leer(ziel)) { console.log(`✓ ${teil} vorhanden`); continue; }
  const quelle = path.join(ALT, teil);
  if (!da(quelle)) {
    console.error(`FEHLT: ${teil} — weder in ${ziel} noch in ${quelle}.`);
    console.error("Erstausstattung: im whisper-web-Checkout `cd electron && npm run build:bundle` laufen lassen (lädt CPython, kopiert whisper-cli/ffmpeg/Modell) — oder die Teile hier von Hand ablegen.");
    process.exit(1);
  }
  kopiere(quelle, ziel);
}

// 2. venv frisch (Symlink-venv + relative Links, v1-Muster — Symlinks
//    erhalten den @rpath auf libpython; --copies bräche ihn)
const venv = path.join(RES, "venv");
const py = path.join(RES, "python-runtime/bin/python3");
if (forceVenv || leer(venv) || !fs.existsSync(path.join(venv, "bin/python3"))) {
  fs.rmSync(venv, { recursive: true, force: true });
  console.log("baue venv …");
  execFileSync(py, ["-m", "venv", venv], { stdio: "inherit" });
  // absolute Symlinks in venv/bin → relativ (Bundle ist relozierbar)
  for (const name of fs.readdirSync(path.join(venv, "bin"))) {
    const p = path.join(venv, "bin", name);
    const st = fs.lstatSync(p);
    if (!st.isSymbolicLink()) continue;
    const ziel = fs.readlinkSync(p);
    if (!path.isAbsolute(ziel)) continue;
    const rel = path.relative(path.dirname(p), ziel);
    fs.rmSync(p); fs.symlinkSync(rel, p);
  }
  const pip = path.join(venv, "bin/pip");
  if (!da(ENRICH_CORE)) {
    console.error(`enrich-core fehlt (${ENRICH_CORE}) — der .enrich-Export braucht es.`);
    process.exit(1);
  }
  execFileSync(pip, ["install", "--upgrade", "pip"], { stdio: "inherit" });
  execFileSync(pip, ["install", ENRICH_CORE, path.join(ROOT, "backend")],
               { stdio: "inherit" });
  // Tauris Resource-Bundler DEREFERENZIERT Symlinks: venv/bin/python3
  // wird im .app eine echte Datei, deren @rpath libpython3.13.dylib in
  // venv/lib/ sucht (Live-Befund 2026-08-30) — die dylib liegt deshalb
  // zusätzlich dort.
  fs.mkdirSync(path.join(venv, "lib"), { recursive: true });
  fs.copyFileSync(path.join(RES, "python-runtime/lib/libpython3.13.dylib"),
                  path.join(venv, "lib/libpython3.13.dylib"));
  // Smoke-Test
  execFileSync(path.join(venv, "bin/python3"),
    ["-c", "import localtranscript.main, enrich_core, fitz, torch, speechbrain, silero_vad, sklearn; print('venv ok')"],
    { stdio: "inherit" });
} else {
  // venv steht — aber unser Backend-Code ändert sich laufend:
  // localtranscript + enrich-core IMMER frisch einspielen (billig)
  console.log("✓ venv vorhanden — aktualisiere localtranscript + enrich-core");
  execFileSync(path.join(venv, "bin/pip"),
    ["install", "--force-reinstall", "--no-deps", "-q",
     ENRICH_CORE, path.join(ROOT, "backend")], { stdio: "inherit" });
}

// 3. Marker
fs.writeFileSync(path.join(RES, "BUNDLED"),
                 `LocalTranscript bundle ${new Date().toISOString()}\n`);
console.log("Resources bereit:", RES);
