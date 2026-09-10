# Changelog

All notable changes to LocalTranscript. The GitHub release notes for
a version are the corresponding section of this file.

## Unreleased

### Added

- **Identity in the dossier** (Settings). An optional e-mail address as
  the app's user ID: it is written into every enrich dossier you export
  — as the person in the journal of who edited what and when — and
  leaves the computer only inside the file you pass on yourself. The
  settings say so next to the field. Alongside it a random
  installation ID (`ins-…`), generated on first launch, shown and
  regenerable in Settings; it tells two installations apart in a
  journal without naming a device or a person. Never a hardware UUID or
  hostname.

### Changed

- **The enrich dossier is one file: `.enrich`.** The export no longer
  writes `.enrich.zip` but `<name>.enrich` — a zip container like
  `.docx` or `.qdpx`, stored without compression (the audio inside does
  not compress anyway). enrich opens it by content. On macOS the app
  declares the dossier type, so a `.enrich` shows as a single document
  whether or not enrich is installed.
- **Import takes the dossier in both forms:** the `.enrich` file, and
  the dossier folder as enrich keeps it while working (a package on
  macOS) — chosen in the dialog or dropped onto the Human Editor list.
  Files named `.enrich.zip` by earlier versions are still read; they are
  just no longer produced or offered.
- Drag and drop onto the Human Editor list imports `.enrich`, `.vtt`
  and `.csv` files.

## 2.2.0 — 2026-09-09

Requires macOS 14 or later, Apple Silicon. Signed with a Developer ID
and notarised by Apple; the notarisation tickets are stapled to both
the disk image and the app.

### Added

- **REFI-QDA export (`.qdpx`)** for ATLAS.ti, NVivo and MAXQDA. The
  transcript is written both as a `TextSource` and as the
  `Transcript` of an `AudioSource`, with one `SyncPoint` per utterance
  (character offset ↔ millisecond). Speakers become codes; each
  utterance becomes a coded selection. The audio is placed in a
  `<name> Media/` folder alongside the `.qdpx`.
- **Transcript handover through `.enrich.zip`.** The dossier now also
  carries `transkript.json`, the canonical model. The Human Editor
  import accepts `.zip` and restores transcript and audio unchanged;
  analysis layers are discarded. Dossiers without the file are read
  from the time map at turn granularity.
- **Find and replace** as a second tab in the side panel. Literal
  matching, no regular expressions. Options for case sensitivity and
  for hyphenation (finds `Werk- statt` for `Werkstatt`). Replace,
  skip, replace all.
- **Batch queue** with a setting for simultaneous runs (Settings →
  Standard options, default 1). Files are processed in the order they
  were dropped; waiting runs are marked as such.
- **Elapsed and estimated time** per run in the batch list, projected
  from actual progress.
- **About dialog** with version and links to source code, releases,
  licence text and the bundled tools.
- **Formats card** in Settings listing the interchange formats and the
  licences of their specifications.

### Changed

- **Licence: AGPL-3.0-or-later** (previously GPL-3.0-or-later).
  PyMuPDF (AGPL-3.0) typesets the dossier PDF in the enrich export;
  GPLv3 §13 permits the combination but extends the AGPL network
  clause to the work as a whole. The backend binds to `127.0.0.1`
  only and rejects foreign hosts, and the source is public.
- **Backend port 5628** (previously 44100), overridable with
  `LT_SERVE_PORT`. The browser interface is at
  <http://127.0.0.1:5628/>.
- **Speaker count** offers exact values 1–6 and Automatic instead of
  overlapping ranges. Selecting 1 disables speaker detection.
- **Speaker detection reports progress**; cancelling now takes effect
  immediately instead of after the full detection pass.
- **Window title** appears once instead of twice.
- Icon set unified on one size scale; 16 px layout grid throughout the
  interface.
- Transport controls centred; timecodes use tabular figures.

### Fixed

- Arrow key navigation between segments while playback is paused.
- Text field and row actions overlapped in the segment list.
- Simultaneous speaker detection could terminate the backend process.
- A second instance could terminate the first instance's backend.
- Baselines within a segment row were misaligned.

### Bundled components

whisper.cpp (MIT) · large-v3-turbo model (OpenAI, MIT) · silero-vad
(MIT) · SpeechBrain ECAPA (Apache-2.0) · PyMuPDF (AGPL-3.0) ·
Recursive typeface (SIL OFL 1.1) · FastAPI/uvicorn (MIT) ·
React/Radix (MIT) · Lucide (ISC) · ffmpeg 9.0.1, GPL static build from
<https://ffmpeg.martin-riedl.de> (source:
<https://ffmpeg.org/download.html>) — the GPL §6 source offer.

The REFI-QDA specification is MIT-licensed, Copyright 2019 REFI-QDA.
LocalTranscript supports export to REFI-QDA; this is not an official
certification, and the MIT licence of the specification does not cover
trademarks.

## 2.0.0 — 2026-08-30

Complete rebuild as a Tauri app (previously Electron): fully offline
transcription with speaker diarisation, a library model, a segment
editor and the enrich export.

- Three tabs: AI Transcript (drop, options, batch list), Human Editor
  (library and VTT/CSV import), Settings (de/en/fr/it).
- Canonical storage: one folder per transcript with `transkript.json`
  (speakers as entities), history snapshots on every save, trash
  instead of deletion. VTT/CSV/TXT are derived exports.
- Editor: autosave, rename/merge/assign speakers with audio samples,
  split/join/delete segments, player with follow mode, timecodes
  always hh:mm:ss.
- Exports: VTT (standard voice tags), CSV, TXT and `.enrich.zip`.

### Withdrawn releases

Versions 2.1.0 through 2.1.3 were published and withdrawn. Their app
bundles carried an incomplete code signature: the `.app` had no
`_CodeSignature/` seal, which macOS reports as “damaged”. The tags
remain in the repository; the disk images are no longer available.
Their changes are contained in 2.2.0.
