"""Sprechertrennung über SpeakerKit (Argmax, MIT) auf Core ML.

Das Modell ist **pyannote community-1** (pyannote.audio 4): Powerset-
Segmentierung mit Überlappungs-Erkennung, WeSpeaker-Embeddings, PLDA/VBx-
Clustering — als Core ML auf der Neural Engine. Gerufen wird die
mitgelieferte Kommandozeile `argmax-cli diarize`; sie bekommt die
Modelle aus einem lokalen Ordner (`--model-path`) und lädt nie etwas
nach.

Warum ein eigener Prozess und nicht Python (User-Entscheid 2026-09-13):
- Qualität. Die alte Kette (silero-VAD + SpeechBrain ECAPA + AHC) kannte
  keine Überlappung und verzählte sich; an einer 5-min-Feldaufnahme fand
  sie 8 Sprecher, SpeakerKit 3 (automatisch) bzw. exakt die vorgegebene
  Zahl.
- Tempo. Dieselbe Aufnahme: 8,4 s gegen 0,8 s.
- Grösse. Damit fallen torch, torchaudio, SpeechBrain, silero-vad und
  scikit-learn aus dem Bundle — rund 750 MB.

Die öffentliche Schnittstelle bleibt, wie sie war:
  diarize_audio(audio_path, min_speakers, max_speakers, threshold) -> list[SpeakerSegment]
  get_speaker_at_time(segments, time) -> str | None
  SpeakerSegment(start, end, speaker)
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import get_argmax_cli, get_speakerkit_dir


@dataclass
class SpeakerSegment:
    start: float  # Sekunden
    end: float    # Sekunden
    speaker: str  # Rohlabel der Diarisierung, z. B. "A"


def _schwelle(ui: float) -> float:
    """Unsere Trennschärfe (0,25 streng … 0,7 locker) auf die
    VBx-Distanzschwelle der CLI abbilden (Standard dort 0,6).

    Beide Skalen laufen gleichsinnig — kleiner heisst mehr Sprecher —,
    nur der Bereich ist enger: 0,25 → 0,45 · 0,5 → 0,62 · 0,7 → 0,75.
    """
    ui = max(0.25, min(0.7, ui))
    return round(0.45 + (ui - 0.25) * (0.30 / 0.45), 3)


def rttm_lesen(text: str) -> list[SpeakerSegment]:
    """RTTM → Segmente, zeitlich sortiert und überlappungsfrei.

    Die CLI läuft mit `--use-exclusive-reconciliation`, gibt also je
    Zeitpunkt nur einen Sprecher aus. Die Klammer hier bleibt trotzdem:
    der Job schneidet aus jedem Block einen Audioclip, und überlappende
    Blöcke hiessen doppelt transkribierter Text.
    """
    roh: list[SpeakerSegment] = []
    for zeile in text.splitlines():
        f = zeile.split()
        if len(f) < 8 or f[0] != "SPEAKER":
            continue
        try:
            start, dauer = float(f[3]), float(f[4])
        except ValueError:
            continue
        if dauer <= 0:
            continue
        roh.append(SpeakerSegment(start, start + dauer, f[7]))
    roh.sort(key=lambda s: (s.start, s.end))
    aus: list[SpeakerSegment] = []
    for s in roh:
        if aus and s.start < aus[-1].end:
            if s.end <= aus[-1].end:
                continue                     # ganz verdeckt
            s = SpeakerSegment(aus[-1].end, s.end, s.speaker)
        aus.append(s)
    return aus


def diarize_audio(
    audio_path: str,
    min_speakers: int = 0,
    max_speakers: int = 0,
    threshold: float = 0.5,
    fortschritt=None,
    register=None,
) -> list[SpeakerSegment]:
    """Eine Audiodatei diarisieren.

    `min_speakers == max_speakers > 0` heisst «genau so viele» und wird
    an `--num-speakers` durchgereicht; sonst entscheidet das Clustering.
    `register` bekommt den Prozess, damit «Abbrechen» ihn killen kann.
    """
    cli = get_argmax_cli()
    modelle = get_speakerkit_dir()
    if fortschritt is not None:
        fortschritt(0, 1)
    with tempfile.TemporaryDirectory(prefix="lt-diar-") as td:
        rttm = Path(td) / "aus.rttm"
        cmd = [cli, "diarize", "--audio-path", str(audio_path),
               "--model-path", str(modelle), "--rttm-path", str(rttm),
               "--cluster-distance-threshold", str(_schwelle(threshold)),
               "--use-exclusive-reconciliation"]
        if min_speakers > 0 and min_speakers == max_speakers:
            cmd += ["--num-speakers", str(min_speakers)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        if register:
            register(proc)
        ausgabe, _ = proc.communicate()
        if register:
            register(None)
        if proc.returncode != 0 or not rttm.is_file():
            letzte = (ausgabe or "").strip().splitlines()[-3:]
            raise RuntimeError("Sprechertrennung fehlgeschlagen: "
                               + " / ".join(letzte))
        segmente = rttm_lesen(rttm.read_text("utf-8"))
    if fortschritt is not None:
        fortschritt(1, 1)
    return segmente


def get_speaker_at_time(segments: list[SpeakerSegment],
                        time: float) -> str | None:
    for s in segments:
        if s.start <= time < s.end:
            return s.speaker
    return None
