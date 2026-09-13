#!/usr/bin/env node
// SpeakerKit beschaffen: die Swift-Kommandozeile bauen und die
// Core-ML-Modelle laden — beides landet im Checkout (bin/,
// models/speakerkit) und wird von bundle-resources.mjs ins Bundle
// kopiert. Läuft EINMAL pro Entwicklerrechner; die App selbst lädt
// nie etwas nach.
//
//   node scripts/hole-argmax.mjs [--force]
//
// Quellen:
//   Code    argmaxinc/argmax-oss-swift (MIT), Commit unten angeheftet
//   Modelle argmaxinc/speakerkit-coreml auf Hugging Face — pyannote
//           community-1 als Core ML (Upstream CC-BY-4.0)
// Voraussetzung: Xcode-Toolchain (`swift --version`), Netz.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const COMMIT = "ea872ff";                       // geprüfter Stand 2026-09-13
const REPO = "https://github.com/argmaxinc/argmax-oss-swift.git";
const HF = "https://huggingface.co/argmaxinc/speakerkit-coreml/resolve/main";
const API = "https://huggingface.co/api/models/argmaxinc/speakerkit-coreml/tree/main";
// Nur die vier Varianten, die die CLI tatsächlich lädt (16 MB statt 33):
const VARIANTEN = ["speaker_segmenter/pyannote-v3/W32A32",
                   "speaker_segmenter/pyannote-v3/W8A16",
                   "speaker_embedder/pyannote-v3/W8A16",
                   "speaker_clusterer/pyannote-v4/W32A32"];
const force = process.argv.includes("--force");
const CLI = path.join(ROOT, "bin/argmax-cli");
const MODELLE = path.join(ROOT, "models/speakerkit");

async function dateiliste(prefix) {
  const r = await fetch(`${API}/${prefix}?recursive=true`);
  if (!r.ok) throw new Error(`Hugging Face antwortet ${r.status} für ${prefix}`);
  return (await r.json()).filter(e => e.type === "file").map(e => e.path);
}

// 1. CLI bauen
if (force || !fs.existsSync(CLI)) {
  const bau = fs.mkdtempSync(path.join(os.tmpdir(), "argmax-"));
  console.log(`baue argmax-cli (${COMMIT}) in ${bau} …`);
  execFileSync("git", ["clone", "--quiet", REPO, bau], { stdio: "inherit" });
  execFileSync("git", ["-C", bau, "checkout", "--quiet", COMMIT], { stdio: "inherit" });
  execFileSync("swift", ["build", "-c", "release", "--product", "argmax-cli"],
               { cwd: bau, stdio: "inherit" });
  fs.mkdirSync(path.join(ROOT, "bin"), { recursive: true });
  fs.copyFileSync(path.join(bau, ".build/release/argmax-cli"), CLI);
  fs.chmodSync(CLI, 0o755);
  fs.rmSync(bau, { recursive: true, force: true });
  console.log("✓ bin/argmax-cli");
} else {
  console.log("✓ bin/argmax-cli vorhanden (--force baut neu)");
}

// 2. Modelle laden
let geladen = 0;
for (const v of VARIANTEN) {
  for (const rel of await dateiliste(v)) {
    const ziel = path.join(MODELLE, rel);
    if (!force && fs.existsSync(ziel)) continue;
    fs.mkdirSync(path.dirname(ziel), { recursive: true });
    const r = await fetch(`${HF}/${rel}`);
    if (!r.ok) { console.error(`FEHLER ${r.status}: ${rel}`); process.exit(1); }
    fs.writeFileSync(ziel, Buffer.from(await r.arrayBuffer()));
    geladen++;
  }
}
console.log(`✓ models/speakerkit (${geladen} Datei(en) geladen)`);

// 3. Probe: läuft die CLI offline?
const hilfe = execFileSync(CLI, ["diarize", "--help"], { encoding: "utf8" });
if (!hilfe.includes("--num-speakers")) {
  console.error("ABBRUCH: argmax-cli kennt --num-speakers nicht — Commit prüfen.");
  process.exit(1);
}
console.log("SpeakerKit bereit.");
