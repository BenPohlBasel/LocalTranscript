# LocalTranscript — Description of the data processing

A text block to paste into a record of processing activities, a data
protection impact assessment, an ethics application or a data
management plan. As of 10 September 2026, LocalTranscript 2.2.0. Items
in `[square brackets]` are completed by the controller.

The text describes what the software does and does not do. The legal
assessment of the controller's own processing — under the GDPR or the
revised Swiss FADP — is made by the controller; the text is no
substitute for legal advice.

---

## 1. Software used

LocalTranscript, version `[2.2.0]`. Free software under
AGPL-3.0-or-later, developed at B/IAS – Basel Institut für angewandte
Stadtforschung. Source code public at
<https://github.com/BenPohlBasel/LocalTranscript>. The software runs as
a local application on macOS (Apple Silicon) and is installed and
operated by the controller itself.

## 2. Purpose of the processing

Conversion of audio recordings `[e.g. semi-structured interviews in the
project …]` into text with timecodes and speaker attribution, for
subsequent qualitative analysis `[in ATLAS.ti / MAXQDA / NVivo /
enrich / …]`.

## 3. Data subjects and categories of data

The data subjects are the recorded persons `[interviewees, participants
in group discussions, …]`. Processed are voice recordings (voice and
content of the conversation) and the transcripts generated from them,
with timecodes and speaker attribution. Depending on the content of the
conversation, special categories of personal data may be involved
`[yes / no: …]`.

## 4. Data flow of one transcription

1. **Input.** The audio file is read from the local file system of the
   device (MP3, WAV, M4A, OGG, FLAC).
2. **Processing.** Speech recognition (whisper.cpp, model
   large-v3-turbo) and speaker separation (silero-vad, SpeechBrain
   ECAPA) run inside the application's process on the device's
   processor or graphics unit. All models are contained in the
   application package; nothing is downloaded on first launch.
3. **Storage.** For each transcript a folder is created at the chosen
   location `[path, e.g. ~/Documents/LocalTranscript]` holding a copy of
   the audio, the canonical transcript file (JSON), history snapshots
   written on every save, and the derived exports. Temporary working
   files are removed after each run.
4. **Network.** The software opens no outbound network connections: no
   telemetry, no usage statistics, no update check, no crash reports of
   its own. The application's internal service binds exclusively to the
   loopback address `127.0.0.1` and rejects requests from any other host
   (HTTP 421). Diagnostic data of the macOS operating system is governed
   by its system settings, not by the software.
5. **Output.** Export files (WebVTT, CSV, plain text, REFI-QDA `.qdpx`,
   enrich dossier `.enrich.zip`) are written wherever the operator saves
   them. **REFI-QDA and enrich exports contain the audio recording.**
   Passing them on is passing on the recording.

## 5. Location of the processing

Exclusively on the device `[device, location]` in the session of the
logged-in user. There is no server, no cloud service and no user
account.

## 6. Recipients, processors, transfers to third countries

None. Since no data is transmitted, there are neither recipients nor
processors nor a transfer to a third country. Data leaves the device
only if the controller itself passes on export files `[to …, by …]`.

## 7. Retention and deletion

Retention of recordings and transcripts: `[period, basis]`. Deleting in
the application moves an entry to a trash folder inside the library
(`_papierkorb`); it is removed for good only when that folder is emptied
`[by whom, when]`. History snapshots live in the folder of the
respective transcript and are deleted with it. Backups of the device
`[Time Machine, …]` are subject to the controller's deletion rule.

## 8. Technical and organisational measures

To be provided by the controller, as the software itself brings no
access control:

- Disk encryption, e.g. FileVault: `[active since …]`
- Access protection of the device (login, screen lock): `[…]`
- Pseudonymisation before any transfer — in the application's editor,
  speakers can be renamed and names in the text swapped with find and
  replace; the decision of what to replace is made by the editing
  person: `[procedure, responsibility]`
- Backup rule: `[…]`
- Rule for passing on export files, in particular those containing
  audio: `[…]`

## 9. Legal basis and information of the data subjects

`[consent / legitimate interest / research privilege under …;
information letter of …]`. The software contributes nothing here.

## 10. Verifiability

The statements in section 4 can be checked against the source code: the
binding of the internal service to `127.0.0.1` and the rejection of
foreign hosts are in `backend/src/localtranscript/main.py`. The
repository contains the complete build chain up to the signed
installation package; anyone who does not trust the distributed binary
can build it.

---

Source of this text: <https://github.com/BenPohlBasel/LocalTranscript>
(folder `site/docs`). It may be used and adapted freely.
