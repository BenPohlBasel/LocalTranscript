# LocalTranscript 2.0

Native macOS-App für **vollständig lokale** Audio-Transkription mit
Sprecher-Diarisierung — Neubau des LocalTranscript-Prototyps als
**Tauri**-App mit enrich-Kit-Oberfläche und **enrich-Export**.

- **Transkription:** [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
  (Metal), Modell `large-v3-turbo` im Bundle.
- **Diarisierung:** silero-vad + SpeechBrain ECAPA-TDNN (kein Token nötig).
- **Bibliothek statt Datei-Chaos:** je Transkript EIN Ordner mit
  kanonischem `transkript.json` (Sprecher als Entitäten, Segmente mit
  IDs), `history/`-Snapshots bei jedem Speichern, Papierkorb statt
  Löschen. VTT/CSV/TXT sind **abgeleitete Exporte**, nie mehr Quelle.
- **Editor:** Autosave, Sprecher umbenennen/zusammenführen/umhängen,
  Hörprobe je Sprecher, Segmente teilen/verbinden/löschen,
  Audio-Player mit Folgen-Modus, Timecodes immer hh:mm:ss.
- **Exporte:** VTT · CSV · TXT · **`.qdpx`** (REFI-QDA, für ATLAS.ti /
  NVivo / MAXQDA: Transkript als TextSource *und* als `Transcript`
  einer `AudioSource` mit einem `SyncPoint` je Äußerung, Sprecher als
  Codes) · **`.enrich.zip`** — ein vollwertiges
  [enrich](https://github.com/BenPohlBasel/PDFenrichCLI)-Dossier
  (T0=T1 byte-treu, PDF in Recursive gesetzt, Zeitkarte
  Offsets↔Sekunden↔Sprecher, Audio-Kopie, Analyse-Kette „narrativ") —
  direkt in enrich importierbar.
- **Weitergabe:** dasselbe `.enrich.zip` kann beides — enrich füttern
  und ein Transkript an Kolleginnen und Kollegen geben, die im
  Human-Editor weiterarbeiten. Im Dossier liegt `transkript.json` als
  Beilage (die kanonische Wahrheit, ein VTT als JSON), das Audio ist
  ohnehin dabei. Der Import zieht **nur** diese beiden — jede
  Analyse-Schicht fällt weg, denn nach dem ersten Edit stimmt keine
  davon mehr.
- **Privat:** Loopback only (127.0.0.1:5628 — „LOCT" auf der
  Telefontastatur; `LT_SERVE_PORT` überschreibt), keine Netz-Calls.

## Architektur

```
backend/    Python ≥3.12 (uv): FastAPI, whisper-cli-Wrapper,
            Diarisierung, Bibliothek, Exporte
            └─ src/localtranscript/enrich_export/  gevendorte
               enrich-Bausteine (Drift-Guard-Test) + Recursive-Fonts (OFL)
frontend/   React 18 + TS + Vite, Radix Themes, enrich-Kit
            (components/ui.tsx), i18n de/en/fr/it
            └─ src-tauri/   Shell: spawnt das Backend, Save-Dialoge
scripts/    bundle-resources.mjs (python-runtime, whisper-cli, ffmpeg,
            Modell, venv → src-tauri/resources)
```

Die Shell ist dumm: Config, Bibliothek und Jobs besitzt das Backend.
Ein im Terminal gestartetes Backend wird von der App benutzt, nie
angefasst; beendet wird nur, was die App selbst gestartet hat.

## Entwicklung

Voraussetzungen: macOS Apple Silicon, uv, Node 18+, Rust/cargo,
`brew install whisper-cpp ffmpeg`, Modell `ggml-large-v3-turbo.bin`
unter `models/` oder `~/whisper-models/` (oder `LT_MODELS_DIR`).
**enrich-Checkout als Geschwister** (`../enrich`) — enrich-core ist
Pfad-Dependency des .enrich-Exports.

```bash
cd backend && uv sync && uv run pytest          # Backend + Tests
cd frontend && npm install
npm run dev                                      # Browser-Dev (Proxy :5628)
uv run uvicorn localtranscript.main:app --port 5628    # in backend/
npx tauri dev                                    # App-Dev
```

## Bundle bauen

```bash
node scripts/bundle-resources.mjs   # Runtime/Binaries/Modell + venv
node scripts/sign-resources.mjs     # 250 Programmteile: Developer ID
cd frontend && npx tauri build      # .app + .dmg, Hülle versiegelt
```

**Die Reihenfolge ist Pflicht.** Tauri signiert nur die Hülle und das
Hauptprogramm; die 250 mitgelieferten Bibliotheken (python3,
whisper-cli, ffmpeg, torch, SpeechBrain) behalten sonst die
Ad-hoc-Signatur des Linkers — und die lehnt Apples Notardienst ab,
nach dem 1,9-GB-Upload. `sign-resources.mjs` signiert sie mit
Developer ID, Hardened Runtime, Entitlements und Zeitstempel (55 s);
die Signatur liegt im Mach-O und überlebt das Kopieren ins Bundle.

Die Berechtigungen in `frontend/src-tauri/entitlements.plist` sind
nicht verhandelbar: ohne `disable-library-validation` startet die App
nach dem Signieren nicht, weil Python .so-Dateien mit fremder
Signatur lädt.

Alles zusammen in EINEM Aufruf (braucht ein app-spezifisches Passwort
von appleid.apple.com):

```bash
export APPLE_ID=…  APPLE_PASSWORD=…  APPLE_TEAM_ID=CCRJ4A42D3
cd frontend && npm run release
spctl -a -vv /Applications/LocalTranscript.app   # → Notarized Developer ID
```

`npm run release` kettet die vier Schritte: Resources einspielen,
Programmteile signieren, bauen (Tauri notarisiert die .app) und
zuletzt `notarize-dmg.mjs` — denn **Tauri notarisiert das DMG nicht
mit**. Ohne diesen Schritt meldet Gatekeeper beim Doppelklick auf das
geladene Image „Unnotarized Developer ID", obwohl die App darin
sauber ist. Hängt das Ticket schon, tut das Skript nichts und spart
den 1,8-GB-Upload.

`bundle-resources.mjs` übernimmt python-runtime, whisper-cli/dylibs
und Modelle aus einem daneben liegenden v1-Checkout
(`../whisper-web/electron/resources`) und baut das venv frisch.
**ffmpeg**: redistributabler GPL-Static-Build von
<https://ffmpeg.martin-riedl.de> (macos/arm64/release) →
`frontend/src-tauri/resources/bin/ffmpeg`; das Skript verweigert
nonfree-Builds (der v1-Binary erklärte sich selbst als „not legally
redistributable"). Für GPL-§6-Compliance die Build-/Quell-Links dem
GitHub-Release beilegen.

## Austauschformate und ihre Lizenzen

**REFI-QDA (`.qdpx`).** LocalTranscript *unterstützt den Export nach
REFI-QDA*. Die Spezifikation steht unter der **MIT-Lizenz, Copyright
2019 REFI-QDA** (<https://www.qdasoftware.org/>). Es gibt keine
offizielle Zertifizierung für REFI-QDA, und Markenrechte deckt die
MIT-Lizenz der Spezifikation nicht ab — entsprechende Behauptungen
werden hier bewusst nicht erhoben. Die REFI-Schemas (XSD) liegen
**nicht** im Bundle: LocalTranscript schreibt nach der Spezifikation
und verweist nur auf die Schema-Adresse, damit greift die
MIT-Beilagepflicht nicht.

**enrich-Dossier (`.enrich.zip`).** Die Format-Spezifikation steht
unter der **MIT-Lizenz** (BIAS.City).

**WebVTT** (W3C) · **CSV** · **TXT** sind offen und unbeschränkt.

## Lizenz

**AGPL-3.0-or-later** (BIAS.City). Die App übernimmt die Lizenz des
strengsten mitgelieferten Werkzeugs — und das ist **PyMuPDF
(AGPL-3.0)**, nicht ffmpeg. PyMuPDF baut den Dossier-PDF-Satz im
enrich-Export (`enrich_export/textsatz.py`: `fitz.Font`,
`fitz.TextWriter`) und bringt MuPDF als Binärteil mit (58 MB im
Bundle). GPLv3 §13 erlaubt die Verbindung von GPL- und AGPL-Code
ausdrücklich, lässt die AGPL-Netzwerkklausel aber „auf die
Kombination als solche" wirken — deshalb steht das Ganze unter AGPL.

**Die Netzwerkklausel ist erfüllt, bevor sie greift:** LocalTranscript
bindet ausschließlich an `127.0.0.1`, und die Host-Wache in `main.py`
weist alles Fremde mit 421 ab — eine Fernnutzung im Sinne von §13 gibt
es nicht. Der Quellcode liegt ohnehin offen; der Knopf „Quellcode
(GitHub)" im Über-Dialog ist das Angebot in der App selbst.

Mitgeliefert u. a.: whisper.cpp (MIT), Modell large-v3-turbo (OpenAI,
MIT), silero-vad (MIT), SpeechBrain ECAPA (Apache-2.0), PyMuPDF
(AGPL-3.0), Recursive-Schrift (SIL OFL 1.1,
`backend/src/localtranscript/enrich_export/fonts/LICENSE-OFL.txt`),
FastAPI/uvicorn (MIT), React/Radix (MIT), Lucide (ISC), ffmpeg
(GPL-3.0-Build, `--enable-gpl --enable-version3`) — vollständige
Liste in den Einstellungen.
