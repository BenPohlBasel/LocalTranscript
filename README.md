# LocalTranscript

Native macOS app for **fully offline** audio transcription with speaker diarization. Drop in audio files, get back VTT subtitles, plain text and CSV — every byte stays on your Mac.

- **Transcription:** [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with Apple Metal GPU acceleration, OpenAI's `large-v3-turbo` model bundled.
- **Speaker diarization:** [silero-vad](https://github.com/snakers4/silero-vad) + [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) with cosine-distance clustering. **No HuggingFace token required.**
- **Shell:** Electron with a small Python (FastAPI + uvicorn) backend spawned on a free local port.
- **Privacy:** loopback only (`127.0.0.1`), no network calls during transcription, all data stays on disk under your chosen folder.

## Status

This is a working prototype rebuilt from a previous py2app/PyWebView app. macOS Apple Silicon only, tested on macOS 14+.

## Install (end users)

1. Download the latest `LocalTranscript-x.y.z-arm64.dmg` from the [Releases page](https://github.com/BenPohlBasel/LocalTranscript/releases).
2. Mount, drag `LocalTranscript.app` into `/Applications`.
3. First launch: **right-click → Open → confirm** (Gatekeeper, the build is not Apple-notarized).
4. The app asks once where transcripts should be stored. Default is `~/Documents/LocalTranscript`. Each batch creates a `yymmdd_hhmm/` subfolder with the original audio plus VTT, TXT and CSV.

The `.dmg` ships everything: standalone CPython 3.13, a venv with PyTorch, SpeechBrain and silero-vad, the Whisper model (≈1.5 GB), `whisper-cli` with all dylibs, and a static ffmpeg. Total ≈1.8 GB.

## Build from source (developers)

### Prerequisites

- macOS Apple Silicon
- [Homebrew](https://brew.sh/) with `whisper-cpp` installed (`brew install whisper-cpp`) — supplies `whisper-cli` and its dylibs
- Node.js 18+ and npm
- Python 3 on `PATH` (only used as the bootstrap to download the standalone CPython for the bundle)
- A copy of the Whisper `large-v3-turbo` model at `models/ggml-large-v3-turbo.bin` (download from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main))

### Dev mode (against your local venv)

```bash
# one-time: create a project venv with the backend deps
python3 -m venv venv
./venv/bin/pip install -r backend/requirements.txt

# install electron deps
cd electron
npm install

# run
npm start
```

The Electron main process spawns `python -m uvicorn backend.main:app` against your local venv on a free port, then loads the UI.

### Build a packaged `.app` and `.dmg`

```bash
cd electron
npm install
npm run build:bundle    # downloads python-build-standalone, creates venv with backend/requirements.txt,
                        # copies whisper-cli + dylibs from Homebrew Cellar, copies ffmpeg-static,
                        # syncs backend/, frontend/, models/ into resources/
npm run dist            # invokes electron-builder, produces dist/LocalTranscript-x.y.z-arm64.dmg
```

To launch the packaged layout from a checkout (without re-running the bundle every time):

```bash
WHISPER_USE_BUNDLE=1 npm start
```

### Layout

```
backend/                 FastAPI + uvicorn server, whisper.cpp wrapper, SpeechBrain diarization
frontend/                vanilla HTML/CSS/JS UI
electron/
├── main.js              spawns the backend, owns the window, native menu, IPC handlers
├── preload.js           contextBridge with first-run config + autosave + shell helpers
├── package.json         electron-builder config (dmg, extraResources)
├── build/icon.icns      app icon
└── scripts/
    ├── build-python-runtime.mjs   downloads CPython 3.13 standalone and creates the venv
    ├── build-binaries.mjs         copies whisper-cli + dylibs + ffmpeg-static into resources/
    └── sync-resources.mjs         mirrors backend/, frontend/, models/, .env into resources/
```

## License

Application code is **MIT** (see [`LICENSE`](LICENSE)). The bundled Python runtime, models and binaries each come with their own licenses; the complete list with upstream links is in [`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md). Required attributions for Apache 2.0 and LGPL components are in [`NOTICE`](NOTICE).

You may redistribute the `.dmg` for free or against donations as long as you ship it together with `LICENSE`, `NOTICE` and `THIRD-PARTY-LICENSES.md` (which the bundle does automatically).

## Acknowledgements

Built on top of:

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) by Georgi Gerganov / ggml.ai
- [OpenAI Whisper](https://github.com/openai/whisper) model weights
- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [silero-vad](https://github.com/snakers4/silero-vad)
- [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/), [FastAPI](https://fastapi.tiangolo.com/), [Electron](https://www.electronjs.org/)
- [astral-sh/python-build-standalone](https://github.com/astral-sh/python-build-standalone) for the relocatable CPython
- [ffmpeg-static](https://github.com/eugeneware/ffmpeg-static) for the LGPL ffmpeg build
