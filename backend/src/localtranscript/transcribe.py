"""whisper.cpp-Aufrufe — nur die LEBENDEN Pfade aus v1 (Token-Level-
Maschinerie war tot und ist nicht portiert).

Zwei Wege:
- transcribe_segment(): ein diarisierter Sprecher-Clip, blockierend,
  JSON-Ausgabe, Zeit-Offset zurückgerechnet.
- transcribe_classic(): ganze Datei ohne Diarisierung, Zeilen-Streaming
  für Fortschritt + Live-Text (progress -1 = nur Text).
Beide nehmen ein optionales Popen-Register für kooperativen Abbruch.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import get_models_dir, get_whisper_cli


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def parse_timestamp(ts: str) -> float:
    """HH:MM:SS.mmm | MM:SS.mmm → Sekunden ("," wird vorher zu ".")."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


def _model_path(model: str) -> Path:
    p = get_models_dir() / f"ggml-{model}.bin"
    if not p.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {p}")
    return p


def _segments_from_json(json_path: Path,
                        offset: float = 0.0) -> list[TranscriptSegment]:
    data = json.loads(json_path.read_text("utf-8"))
    aus = []
    for seg in data.get("transcription", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        a = parse_timestamp(seg["timestamps"]["from"].replace(",", "."))
        b = parse_timestamp(seg["timestamps"]["to"].replace(",", "."))
        aus.append(TranscriptSegment(a + offset, b + offset, text))
    return aus


def transcribe_segment(audio_path: Path, model: str, language: str,
                       time_offset: float = 0.0,
                       register: Callable[[subprocess.Popen | None], None]
                       | None = None) -> list[TranscriptSegment]:
    """Ein Clip, blockierend; Zeitstempel um time_offset verschoben."""
    cmd = [get_whisper_cli(), "-m", str(_model_path(model)),
           "-l", language, "-f", str(audio_path), "-oj"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    if register:
        register(proc)
    try:
        proc.wait()
    finally:
        if register:
            register(None)
    json_path = Path(str(audio_path) + ".json")
    if not json_path.exists():
        return []
    try:
        return _segments_from_json(json_path, time_offset)
    finally:
        json_path.unlink(missing_ok=True)


_RE_PROGRESS = re.compile(r"progress\s*=\s*(\d+)%")
_RE_TEXT = re.compile(
    r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s*(.+)")


def transcribe_classic(audio_path: Path, model: str, language: str,
                       progress_cb: Callable[[int, str, str], None]
                       | None = None,
                       register: Callable[[subprocess.Popen | None], None]
                       | None = None) -> list[TranscriptSegment]:
    """Ganze Datei, natürliche Whisper-Segmente, Fortschritt gestreamt."""
    cmd = [get_whisper_cli(), "-m", str(_model_path(model)),
           "-l", language, "-f", str(audio_path), "-oj",
           "--print-progress"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    if register:
        register(proc)
    partial: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            m = _RE_PROGRESS.search(line)
            if m and progress_cb:
                progress_cb(int(m.group(1)), "", " ".join(partial))
            t = _RE_TEXT.search(line)
            if t and t.group(1).strip():
                partial.append(t.group(1).strip())
                if progress_cb:
                    progress_cb(-1, "", " ".join(partial))
        proc.wait()
    finally:
        if register:
            register(None)
    json_path = Path(str(audio_path) + ".json")
    if not json_path.exists():
        return []
    try:
        return _segments_from_json(json_path)
    finally:
        json_path.unlink(missing_ok=True)
