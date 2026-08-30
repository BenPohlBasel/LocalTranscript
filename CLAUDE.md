# LocalTranscript 2.0 (enrich-transcript)

Side-Projekt von enrich (User-Auftrag 2026-08-30): Neubau des
Electron-Prototyps `../whisper-web` als Tauri-App. Plan:
~/.claude/plans/localtranscript-tauri.md. README.md = Architektur.

## Eiserne Regeln (geerbt aus enrich, gelten hier)

- UI NUR über `frontend/src/components/ui.tsx` (enrich-Kit-Kopie);
  Icons nur über `components/icons.tsx`; Strings nur über
  `lib/i18n.ts` — VIERSPRACHIG de/en/fr/it, de = Quellsprache.
- Timecodes IMMER hh:mm:ss (User: „unbedingt").
- `transkript.json` ist die kanonische Wahrheit; Exporte sind
  abgeleitet. Schreiben immer atomar + history/-Snapshot; Löschen =
  `_papierkorb/`, nie destruktiv.
- `backend/src/localtranscript/enrich_export/` ist VENDOR-Code
  (enrich@3d2b131): nie formatieren/fixen (ruff-exclude!), Drift-Guard
  `tests/test_drift_guard.py` vergleicht gegen `../enrich`. Bei
  enrich-Änderungen an textsatz/textimport: neu vendoren (Kopf-
  kommentar sagt wie).
- Die Shell beendet NUR selbst gestartete Backends (ps-identifiziert);
  fremde Prozesse auf Port 44100 sind tabu.
- Recursive-Fonts: SIL OFL 1.1 — Nennung in Einstellungen (steht) und
  bei jeder Veröffentlichung.

## Stand 2026-08-30 (Erstbau, eine Session)

Backend komplett (15 Tests grün, whisper-/ffmpeg-frei über Fakes):
bibliothek/jobs/transcribe/diarize/ausgabe/exporte/main. v1-Bugs
gefixt statt portiert: .wav-Resample-Bypass, nicht-idempotentes
Sprecher-Umbenennen (strukturell weg), Job-Abbruch killt Prozesse.
Frontend komplett (tsc+vite grün, Playwright-verifiziert an echtem
3,4-h-Workshop: 1148 Segmente/8 Sprecher). enrich-Export am selben
Material: 8 s, 73-Seiten-PDF, Zeitkarte, Kette narrativ. Tauri-Shell
kompiliert; Bundle über scripts/bundle-resources.mjs (übernimmt
python-runtime/bin/lib/models aus ../whisper-web, venv frisch).

Bewusst NICHT portiert: HF-Token-Screen (tot), PyWebView (app.py),
~⅔ von merge.py (tote Token-Matching-Architektur), Glossar/NER
(war im UI unerreichbar; Wiederaufnahme möglich — Code in
../whisper-web/backend/glossary.py).

## Test/Dev

`cd backend && uv run pytest` · `uv run ruff check src tests` ·
Frontend `npx tsc && npm run build`. Browser-Demo: uvicorn auf 44100
mit `LT_CONFIG_DIR`-Scratch (Muster in tests/conftest.py).
Echte Transkript-Beispiele: ~/Documents/LocalTranscript/<stamp>/.
