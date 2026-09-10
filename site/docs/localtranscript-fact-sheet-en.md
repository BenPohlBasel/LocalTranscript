# LocalTranscript — Fact sheet for researchers

What the app is, which AI in it does what, where the code lives, and
why the transcript never leaves your computer. For handing to a project
lead, an ethics board or colleagues. As of 10 September 2026, version
2.2.0.

## What the app does

LocalTranscript turns audio recordings — interviews, group discussions,
workshops — into text with timecodes and speaker attribution. The
transcript is then corrected in an editor, speakers are named, names are
replaced, and the result is exported for analysis (ATLAS.ti, MAXQDA,
NVivo via REFI-QDA; enrich; WebVTT, CSV, plain text). All of it happens
on your own Mac.

## Which AI does what

| Component | Task | Origin, licence | Runs where |
|---|---|---|---|
| whisper.cpp with the model `large-v3-turbo` | Speech recognition: audio → text with timecodes | model by OpenAI (MIT), runtime whisper.cpp (MIT) | locally, on the Mac's graphics unit |
| silero-vad | Voice activity: detects where speech occurs at all | MIT | locally |
| SpeechBrain ECAPA-TDNN | Speaker separation: computes voice profiles and clusters them into speakers | Apache-2.0 | locally |

All three models ship inside the application package. There is no
access to an AI service, no account, no API key.

**No generative AI.** Nothing is summarised, rephrased or interpreted.
The app delivers what was said — not what was meant.

**Limits of speech recognition.** Whisper is a neural model. Where it
understands nothing (background noise, dialect, overlapping speech) it
may insert words that were never said. A transcript from
LocalTranscript is a **raw transcript** that must be checked against the
recording; the editor is built for that. Standard German, French,
Italian and English are recognised well, Swiss German patchily — the
speakers are still separated cleanly.

## Where the code lives

- Source code: <https://github.com/BenPohlBasel/LocalTranscript>
- Licence: AGPL-3.0-or-later — free software that may be used,
  inspected, modified and passed on
- Developed at B/IAS – Basel Institut für angewandte Stadtforschung,
  Beckenweg 6, 4056 Basel, <https://bias.city>
- Installer: signed and notarised with an Apple Developer ID, checksum
  with every release on GitHub
- If you do not trust the binary: the repository holds the complete
  build chain, the app can be built from source

## The transcript never leaves the computer

- The app opens **no outbound network connections**: no telemetry, no
  usage statistics, no update check, no crash reports of its own.
- Its internal service listens only on the loopback address `127.0.0.1`
  of the machine itself and rejects any request from elsewhere. The code
  for this is in `backend/src/localtranscript/main.py` — readable by
  anyone.
- Audio and transcript live exclusively in the chosen library folder.
  Deleted items move to a trash folder inside the library until it is
  emptied.
- No server, no cloud service, no user account, no processor, no
  third-country transfer — because nothing is transmitted.

What this does **not** cover: backups of the Mac (Time Machine, iCloud
Drive for the Documents folder) and the diagnostic data of macOS itself
follow the system settings, not the app. Whoever places the library
folder inside a synchronised folder synchronises the recordings.

## Anonymising names

- **Rename speakers:** a name in the speaker panel applies to every
  segment of that person — "Speaker 1" becomes "B3" in one step.
- **Names in the text:** "Find and replace" finds a name in all
  segments, shows each match in context and replaces one at a time or
  all at once — even where the transcript broke the name across a line.
- **What the app does not decide:** which details to replace — places,
  employers, events. That remains the researcher's decision.
- **The recording stays what it is.** Pseudonymisation applies to the
  text. Exports as REFI-QDA (`.qdpx`) and enrich dossier (`.enrich.zip`)
  contain the audio file with voice and real names; WebVTT, CSV and
  plain text contain text only. Whoever wants to pass on pseudonymised
  data only passes on a text format.

## For the methods section

> The recordings were transcribed with LocalTranscript 2.2.0 (B/IAS
> Basel, AGPL-3.0; speech recognition whisper.cpp with the model
> large-v3-turbo, speaker separation with SpeechBrain ECAPA-TDNN)
> entirely locally on a computer of the research group, without
> transmission to external services. The raw transcripts were then
> corrected against the recording and pseudonymised.

---

Source: <https://github.com/BenPohlBasel/LocalTranscript> (folder
`site/docs`). The sheet may be used and adapted freely.
