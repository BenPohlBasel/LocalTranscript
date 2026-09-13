# Changelog

All notable changes to LocalTranscript. The GitHub release notes for
a version are the corresponding section of this file.

## Unreleased

### Changed

- **Fuller attribution for the diarization models.** Argmax's model
  repository now states its licence in its own metadata: the SpeakerKit
  framework is MIT, the models it is built on are CC BY 4.0. That
  licence asks for attribution and for marking changes, so the licence
  list in Settings, the README, the website and the four information
  sheets now name both models, pyannote community-1 and WeSpeaker
  ResNet34, and say that Argmax converted them to Core ML and quantised
  them. The licence list inside the application still carries the
  shorter wording; it follows with a later release.

## 2.5.0 — 2026-09-13

### Changed

- **New speaker separation.** Speakers are now found by pyannote
  community-1 running on the Mac's Neural Engine, through Argmax's
  SpeakerKit command line (MIT). The old chain (silero-VAD plus
  SpeechBrain ECAPA plus agglomerative clustering) knew nothing about
  overlapping speech and miscounted voices. Measured on the same
  five-minute field recording: the old chain invented eight speakers in
  8.4 seconds, the new one finds three in 0.9 seconds, and with a
  speaker count given it returns exactly that many (two speakers, 0.8
  seconds). An hour-long workshop recording is separated in 11.9
  seconds. Nothing is downloaded: the models sit in the application
  package and are read from there.
- **The application package shrinks by about 770 MB.** PyTorch,
  torchaudio, SpeechBrain, silero-vad, scikit-learn and SciPy leave the
  bundle (793 MB); the Core ML models and the command line add 24 MB.
  Voice activity detection for transcription is unaffected. It still
  runs inside whisper.cpp with the bundled Silero model.
- **Licence note.** The bundled weights are pyannote community-1
  (CC-BY-4.0), the runtime is SpeakerKit by Argmax (MIT). Both are
  listed in Settings, on the website and in the four information
  sheets.

### Added

- **Every recording in the queue gets its own speaker count, and
  nothing starts without one.** Dropped files now collect in a list
  instead of starting straight away. Each line carries its own choice,
  and the start button stays locked while one is missing. The count
  shapes the result, so it is written into the transcript's source data
  and journal, where it can be read back later. The setting in the
  options is now only a default for newly added files; dropping several
  files at once leaves the choice open.

### Fixed

- **The voice sample stops the one before it.** Two quick clicks played
  two voices at once. A second click on the same voice now stops it, and
  leaving the speaker panel silences it.
- **Clicking beside a speaker badge no longer opens the speaker menu.**
  The badge button used to fill its whole column, so the empty strip
  between badge and text belonged to it. That strip is now neutral: a
  click there takes the focus out of the text field, which is what the
  up and down arrow keys need to walk the segments again.

## 2.4.3 — 2026-09-12

### Fixed

- **No more invented lines on silence.** Whole-file transcription now
  runs with whisper.cpp's voice-activity detection (Silero VAD v5 as
  ggml, bundled inside the package): passages without speech are
  skipped instead of being filled with “* Musik *” or a sentence that
  repeats eight times. Measured on a five-minute field recording: same
  time, same word count, zero repetition loops instead of eight,
  timestamps unchanged. A clip with no speech at all yields an empty
  transcript. As a safety net, three or more identical consecutive
  segments are collapsed into one.
- **Smooth video playback.** The picture now runs freely at the
  audio's rate and is only re-aligned on play/pause, on a jump, or when
  it drifts more than a second; the former quarter-second corrections
  (a seek every few seconds) made 4K HEVC from an iPhone stutter.
  Measured: 20 s of 4K, zero seeks, zero stalls, zero dropped frames.
- **Short clips no longer crash speaker separation.** A recording with
  a single speech window made the clustering abort (“Found array with
  1 sample(s)”); it is now one speaker.

## 2.4.2 — 2026-09-12

### Changed

- **Loading feedback everywhere.** A spinner in the header turns while
  any request to the backend is running or the editor is building a
  long transcript; the editor shows “n segments being built …” while
  the rows appear.
- **The editor opens long transcripts three times faster.** Row
  heights were measured one row at a time while the list grew (a forced
  layout per row); they are now measured in one batch — 1200 segments
  in 0.6 s instead of 2.1 s in WebKit.
- **Re-importing a dossier keeps the Zotero role selection.** The
  chosen roles are reconstructed from the people who travelled with the
  layer, so “Reload” after an import fetches the same roles.
- Release tooling: the app is notarised outside `tauri build` with a
  two-hour timeout (`scripts/notarize-app.mjs`), the DMG is built and
  signed by `scripts/build-dmg.mjs` — a slow Apple queue no longer
  costs the build.

## 2.4.1 — 2026-09-11

### Changed

- **Side-panel tabs wrap instead of overflowing.** Speakers | Find |
  Metadata keep their full labels; when the column is too narrow the
  buttons wrap onto a second line (the third tab used to run past the
  panel edge).
- **Portrait video is bounded.** The picture under the speakers is
  limited to 45 % of the window height and letterboxed, so a 9:16
  recording no longer pushes the speaker list off the column.

## 2.4.0 — 2026-09-11

### Added

- **enrich-core is a public MIT package.** The dossier reader/writer
  that LocalTranscript bundles — manifest, schemas, journal, naming
  convention, transcript source, Zotero layer — now lives at
  <https://github.com/BenPohlBasel/enrich-core> under the MIT licence,
  like `FORMAT.md`. The app is built against the tagged release
  (v0.1.0), so the source offer of the AGPL app covers everything in
  the bundle. Settings name it in the licence list.

- **Video as a source.** Drop an MP4/MOV/M4V (H.264 or HEVC) into the
  batch list or pick it as the audio of an import: the sound track is
  extracted to mp3 as the working copy for Whisper and the editor, the
  video file itself is stored unchanged in the entry — nothing is
  transcoded, ever. Only limit: 10 GB per file. Other containers and
  codecs (webm/VP9, mkv, AVI, MPEG-2, AV1) are refused with a hint to
  export as H.264/MP4 externally. In the editor the picture sits under
  the speakers in the side panel — muted, without controls; the audio
  player leads and the picture follows (position, play/pause, speed);
  at 2× and during jumps it freezes and blurs until the picture is back.
  Export: REFI-QDA gets a second entry “with video” (`VideoSource`,
  video in the Media folder); the enrich dossier stays audio only.

### Changed

- **Zotero roles are shown as Zotero names them** — `interviewer`,
  `interviewee`, `director`, `castMember`, … in English, for every item
  type alike, no longer translated: they are Zotero's creator types, not
  app text. Any item type can be linked (video, audio, podcast, film,
  document …); the interviewed side (`interviewee`, `guest`,
  `castMember`, `performer`, `presenter`) is never preselected.

## 2.3.0 — 2026-09-11

### Changed

- **The `.enrich` export carries no PDF any more.** FORMAT.md §5 now
  says transcript + audio are a valid input state and the receiving
  application typesets the reading copy itself — enrich does so on
  import. The export shrinks to the transcript layer
  (`source/transcript.json`, origin per record), the audio as mp3 and
  the Zotero layer; PDF typesetting, PyMuPDF (AGPL-3.0) and the
  Recursive fonts leave the bundle (about 60 MB smaller). The app's
  licence stays AGPL-3.0-or-later by choice; the strictest bundled tool
  is now ffmpeg.

### Added

- **Your own Whisper models.** Drop any whisper.cpp model
  (`ggml-*.bin`) into the library's `Modelle` folder and it appears in
  the model list, marked as your own; a file with the same name as a
  bundled model replaces it. Files are checked (ggml magic, finished
  copying) before they are offered; Settings show the folder, open it,
  rescan, and name what was ignored and why. The app never downloads
  models; the journal records which model transcribed a recording.
  A file that iCloud has evicted from the Mac ("Optimise Mac Storage")
  is reported as such and not opened — opening it would block until
  iCloud has fetched it.

- **Metadata from Zotero.** A third tab in the editor's side panel —
  Speakers | Find | Metadata — links a transcript to an item in your
  local Zotero library. Search by title, year, name or citekey
  (interviews ranked first), pick which people go into the transcript
  by role (the interviewee is not preselected, so a pseudonymised
  transcript does not get their name through the back door), link,
  reload, unlink. The snapshot is stored in the transcript as `source`
  and the link is a human run in the journal. In the `.enrich` export
  it becomes the layer `source/zotero.json`; enrich puts "Title · Date ·
  Interviewer · Citekey" on page 1 when it typesets the text. Zotero is
  read only on request, only read-only (`zotero.sqlite` opened
  immutable — Zotero can stay open), only on this computer, and only
  after you enable it in Settings (optional data directory; the usual
  places are searched otherwise). Nothing is sent to Zotero or the
  network.

- **Export and import follow Format 2 of the enrich dossier**
  (`FORMAT.md` in the enrich repository), built on enrich-core's own
  manifest models so enrich reads it. The container is one uncompressed
  `.enrich` file, laid out by enrich's naming convention (FORMAT.md
  Appendix A: `source/`, `text/`) and packed by enrich-core's own
  `handover` profile: the transcript is the source
  (`source/transcript.json`, schema
  `transcript/1.0.0`, registered in inventory and lineage,
  `source.canonical`), the typeset PDF is `rendered`, every layer
  carries the header `kind · id · origin · from · by · did · result`,
  and the manifest is a complete inventory — every file, including the
  audio, with hash and size — plus lineage (`layers`) and a chained
  journal (`runs`). Import verifies the inventory and refuses a
  container whose files no longer match their hashes. Format-1
  dossiers (earlier exports, enrich's own transcript dossiers) are
  still read.
- **Origin on every record.** Each segment and speaker carries
  `origin`: `machine` for what Whisper and the diarisation produced,
  `source` for a VTT/CSV imported from elsewhere, `human` for anything
  a person touched in the editor — text, speaker, boundaries, a name.
  Only the editor sets `human`; imports carry the flags over and never
  flatten them. Older library entries are upgraded on read.
- **Journal.** Every write is a run: the Whisper job (model,
  diarisation), an import, and each editor session (bundled — ten
  minutes of quiet close a run) with the ids of
  the records it changed and the kind of change. Runs name the app, the
  installation id and, if set, the e-mail; they are chained by the hash
  of the preceding entry and travel with the dossier.

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
