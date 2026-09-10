# LocalTranscript

A native macOS app for **fully local** audio transcription with speaker
diarisation. Whisper runs on your machine, the diarisation runs on your
machine, and nothing is ever uploaded. Built at **B/IAS — Basel
Institut für angewandte Stadtforschung** as a [Tauri](https://tauri.app)
app with the enrich UI kit, and it exports straight into
[enrich](https://github.com/BenPohlBasel/PDFenrichCLI) and into
REFI-QDA for ATLAS.ti.

- **Transcription:** [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
  (Metal), `large-v3-turbo` bundled.
- **Diarisation:** silero-vad + SpeechBrain ECAPA-TDNN — no Hugging Face
  token, no account, no network.
- **Interface:** German, English, French, Italian.
- **Loopback only:** `127.0.0.1:5628` (`LOCT` on a phone keypad;
  `LT_SERVE_PORT` overrides). Requests from any other host are rejected
  with 421.

> The screenshots below use an **invented** two-person interview about
> housing cooperatives, synthesised with macOS `say`. No real research
> material is shown anywhere in this repository. The file itself is in
> [`docs/demo/`](docs/demo/) — 2:02 min, two speakers — so the run can be
> reproduced without recording anything first.

## Transcribe

Drop audio files (MP3, WAV, M4A, OGG, FLAC) onto the *AI Transcript*
tab. Each file becomes a job with a live block counter, the text that is
being recognised right now, and elapsed / estimated time.

Files run **one after another** by default — a second GPU job does not
make the first one faster. *Simultaneous runs* in the settings raises
that to at most four; the queue keeps the order in which files were
dropped, and a waiting job can still be cancelled.

![Batch list with one running and one queued file](docs/screenshots/01-batch.png)

The *Speakers* option is an exact count, not a range: `1` means no
separation and everything ends up as one speaker, `2`–`6` force exactly
that many, `Automatic` lets the clustering decide. *Separation* sets how eagerly voices are
split apart.

## A library, not a pile of files

Every transcript is one folder holding the canonical
`transkript.json` — speakers as entities, segments with stable ids —
plus the audio, `history/` snapshots written on every save, and the
derived exports. Deleting moves to a library trash; nothing is
destroyed.

![Library list](docs/screenshots/02-library.png)

## Edit

Click an entry to open the editor. Autosave, speaker rename / merge /
reassign, an audio sample per speaker, split, join and delete segments,
and a player that follows the text. Timecodes are always `hh:mm:ss`.

![Editor with the speaker panel](docs/screenshots/03-editor.png)

Keyboard, outside the text fields: `J` / `K` / `L` shuttle back, pause
and forward the way Audition does — `←` `→` and Space do the same;
`↑` / `↓` walk along the segments and put the playhead on the start of
each one. With `⌥` held, `J` / `K` / `L` also work **while typing**.
`Ctrl+L` toggles the loop, `Ctrl+X` cycles the playback speed.

## Find and replace

The second tab in the side panel searches **literally** — no
translation, no stemming, no fuzzy matching. It counts the occurrences,
rings the segment it is currently on, and shows the match in its
context. *Replace* changes one and moves on, *Skip* leaves one alone,
*Replace all* does the rest.

*Ignore hyphenation* additionally finds words that a transcript broke
across a line, so searching `cooperative` also finds `coope- rative`.

![Find and replace panel](docs/screenshots/04-find-replace.png)

## Export

![Export menu](docs/screenshots/05-export.png)

- **VTT** — WebVTT with standard voice tags `<v Name>`.
- **CSV**, **TXT** — plain derived text.
- **enrich dossier (`.enrich`)** — a complete enrich dossier: T0=T1
  byte-identical, a PDF typeset in Recursive, a time map linking
  offsets ↔ seconds ↔ speakers, a copy of the audio, and the analysis
  chain marked "narrative". Import it into enrich directly.
- **REFI-QDA for ATLAS.ti (`.qdpx.zip`)** — the transcript as a
  `TextSource` *and* as the `Transcript` of an `AudioSource` with one
  `SyncPoint` per utterance, speakers as codes. Verified against a
  project exported from ATLAS.ti 25.

## Import — and the two steps for VTT

*Import transcript* in the Human Editor takes a transcript that was
made elsewhere and turns it into a library entry.

**A `.enrich` is one step.** It is one file — a zip like `.docx` —
that already carries its audio, so the app takes `transkript.json`
and `audio.mp3` out of it and is done. The same works for a dossier
folder as enrich keeps it (a package on macOS): pick it or drop it. Everything else in the dossier is dropped on purpose: after the
first edit no analysis layer would line up with the text any more.

**A `.vtt` or `.csv` is two steps**, because those formats hold text and
timecodes but no sound:

1. **Pick the transcript.** A file dialog opens for `.vtt` / `.csv`
   (and `.zip`).
2. **Pick the matching audio.** A second dialog asks for the `mp3` that
   belongs to it — *Cancel* if there is none. Without audio the entry
   still works; you simply edit text against timecodes and hear
   nothing.

If you skip the second step, the app looks for an audio file sitting
next to the transcript with the same name (`interview.vtt` →
`interview.mp3`) and takes that. So a matching pair in one folder
imports in one step after all.

## Handing a transcript to a colleague

The same `.enrich` does both jobs: it feeds enrich, and it hands a
transcript to someone else who continues in their own Human Editor.
Text and audio travel together in one file, the timecodes stay exact,
and nothing is lost on the way out or back in.

## Settings

![Settings](docs/screenshots/06-settings.png)

Storage location, default model and language, interface language,
simultaneous runs — and the full list of bundled components with their
licences, the interchange formats with theirs, and the links to the
sources. The same texts appear in the *About LocalTranscript* dialog in
the app menu.

## Documentation

- **This file** — what the app is and does, architecture, development,
  building, licences.
- **[`site/`](site/)** — the project website: a static package with no
  cookies, no dependencies and no build step (fonts and screenshots in
  their own folders), carrying the user guide in German, English,
  French and Italian.
- **[CHANGELOG.md](CHANGELOG.md)** — version history. The GitHub release
  notes are the matching section from it; changes belong here first, not
  in the release form.
- **In the app** — Settings and the About dialog carry the storage
  location, formats, bundled tools and their licences, in four
  languages.

## Architecture

```
backend/    Python ≥3.12 (uv): FastAPI, whisper-cli wrapper,
            diarisation, library, exports
            └─ src/localtranscript/enrich_export/  vendored enrich
               building blocks (drift-guard test) + Recursive fonts (OFL)
frontend/   React 18 + TS + Vite, Radix Themes, enrich kit
            (components/ui.tsx), i18n de/en/fr/it
            └─ src-tauri/   shell: spawns the backend, save dialogs
scripts/    bundle-resources.mjs (python runtime, whisper-cli, ffmpeg,
            model, venv → src-tauri/resources)
            sign-resources.mjs, notarize-dmg.mjs
```

The shell is deliberately dumb: config, library and jobs belong to the
backend. A backend already running in a terminal is used as it is and
never touched; the app only shuts down what it started itself.

## Development

Requirements: macOS on Apple Silicon, uv, Node 18+, Rust/cargo,
`brew install whisper-cpp ffmpeg`, and `ggml-large-v3-turbo.bin` under
`models/` or `~/whisper-models/` (or `LT_MODELS_DIR`). An **enrich
checkout as a sibling** (`../enrich`) — enrich-core is a path dependency
of the `.enrich` export.

```bash
cd backend && uv sync && uv run pytest          # backend + tests
cd frontend && npm install
npm run dev                                      # browser dev (proxy :5628)
uv run uvicorn localtranscript.main:app --port 5628    # in backend/
npx tauri dev                                    # app dev
```

## Building the bundle

```bash
node scripts/bundle-resources.mjs   # runtime/binaries/model + venv
node scripts/sign-resources.mjs     # 250 executables: Developer ID
cd frontend && npx tauri build      # .app + .dmg, shell sealed
```

**The order is mandatory.** Tauri signs the shell and the main binary
only; the 250 bundled libraries (python3, whisper-cli, ffmpeg, torch,
SpeechBrain) would otherwise keep the linker's ad-hoc signature — and
Apple's notary service rejects those, after the 1.9 GB upload.
`sign-resources.mjs` signs them with the Developer ID, hardened runtime,
entitlements and a timestamp (55 s); the signature lives inside the
Mach-O and survives being copied into the bundle.

The entitlements in `frontend/src-tauri/entitlements.plist` are not
negotiable: without `disable-library-validation` the app does not start
after signing, because Python loads `.so` files carrying a foreign
signature.

Everything in one call (needs an app-specific password from
appleid.apple.com):

```bash
export APPLE_ID=…  APPLE_PASSWORD=…  APPLE_TEAM_ID=CCRJ4A42D3
cd frontend && npm run release
spctl -a -vv /Applications/LocalTranscript.app   # → Notarized Developer ID
```

`npm run release` chains the four steps: stage the resources, sign the
executables, build (Tauri notarises the `.app`) and finally
`notarize-dmg.mjs` — because **Tauri does not notarise the DMG**.
Without that step Gatekeeper reports "Unnotarized Developer ID" when the
downloaded image is opened, even though the app inside it is clean. If
the ticket is already stapled the script does nothing and saves the
1.8 GB upload.

`bundle-resources.mjs` takes the python runtime, whisper-cli/dylibs and
models from a v1 checkout sitting next to this one
(`../whisper-web/electron/resources`) and builds the venv fresh.
**ffmpeg**: a redistributable GPL static build from
<https://ffmpeg.martin-riedl.de> (macos/arm64/release) →
`frontend/src-tauri/resources/bin/ffmpeg`; the script refuses nonfree
builds (the v1 binary declared itself "not legally redistributable").
For GPL §6 compliance, attach the build and source links to the GitHub
release.

## Interchange formats and their licences

**REFI-QDA (`.qdpx.zip`).** LocalTranscript *supports export to REFI-QDA*.
The specification is under the **MIT licence, Copyright 2019 REFI-QDA**
(<https://www.qdasoftware.org/>). There is no official certification for
REFI-QDA, and the MIT licence of the specification does not cover
trademarks — no such claim is made here. The REFI schemas (XSD) are
**not** bundled: LocalTranscript writes to the specification and only
references the schema URL, so the MIT attribution requirement does not
apply.

**enrich dossier (`.enrich`).** The format is described in
`FORMAT.md` in the enrich repository — Format 1 as it is, Format 2 as a
proposal — under the **MIT licence** (B/IAS).

**WebVTT** (W3C) · **CSV** · **TXT** are open and unrestricted.

## Licence

**AGPL-3.0-or-later** (BIAS.City). The app adopts the licence of the
strictest bundled tool — and that is **PyMuPDF (AGPL-3.0)**, not ffmpeg.
PyMuPDF typesets the dossier PDF in the enrich export
(`enrich_export/textsatz.py`: `fitz.Font`, `fitz.TextWriter`) and brings
MuPDF along as a binary component (58 MB in the bundle). GPLv3 §13
explicitly permits linking GPL and AGPL code, but lets the AGPL network
clause apply "to the combination as such" — so the whole thing is under
the AGPL.

**The network clause is satisfied before it applies:** LocalTranscript
binds to `127.0.0.1` only, and the host guard in `main.py` turns
everything foreign away with 421 — there is no remote use in the sense
of §13. The source is public anyway; the "Source code (GitHub)" button
in the About dialog is the offer inside the app itself.

Bundled, among others: whisper.cpp (MIT), large-v3-turbo model (OpenAI,
MIT), silero-vad (MIT), SpeechBrain ECAPA (Apache-2.0), PyMuPDF
(AGPL-3.0), Recursive typeface (SIL OFL 1.1,
`backend/src/localtranscript/enrich_export/fonts/LICENSE-OFL.txt`),
FastAPI/uvicorn (MIT), React/Radix (MIT), Lucide (ISC), ffmpeg
(GPL-3.0 build, `--enable-gpl --enable-version3`) — the complete list is
in the settings.
