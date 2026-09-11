"""Video als Quelle (BACKLOG 8, User-Entscheide 2026-09-11).

Keine Umwandlung, keine Grenze bei Dauer oder Auflösung — nur die
Dateigrösse (10 GB). Angenommen wird, was der WKWebKit nativ spielt:
H.264 und HEVC in MP4/MOV/M4V. Alles andere (VP8/VP9/webm, mkv,
AVI/DivX/WMV, MPEG-2, AV1) wird abgewiesen, mit dem Hinweis, extern als
H.264/MP4 zu exportieren. Der Ton wird nach mp3 gezogen (Arbeitskopie
für Whisper und den Editor), das Video liegt unverändert im Eintrag.

Bild und Ton bleiben getrennt: der Editor spielt das mp3, das stumme
Video folgt ihm. enrich bekommt nur den Ton («enrich hat nur Ton»),
REFI-QDA wahlweise das Video.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .config import get_ffmpeg_cli

#: Container, die der WKWebKit nativ spielt
VIDEO_ENDUNGEN = (".mp4", ".m4v", ".mov")
#: Codecs dazu — Hardware-dekodiert auf jedem Apple-Silicon-Mac
CODECS_OK = frozenset({"h264", "hevc"})
#: Einzige Grenze: die Dateigrösse (User 2026-09-11)
VIDEO_MAX_BYTES = 10 * 1024 ** 3

_STREAM = re.compile(r"Stream #\d+:\d+.*?: (Video|Audio): (\w+)")
_MASSE = re.compile(r", (\d{2,5})x(\d{2,5})[, ]")
_DAUER = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")


class VideoFehler(Exception):
    """Abgewiesen — `code` ist maschinenlesbar, `str()` der Hinweis."""

    def __init__(self, code: str, hinweis: str) -> None:
        super().__init__(hinweis)
        self.code = code


def sondiere(pfad: Path) -> dict:
    """Was steckt in der Datei? `ffmpeg -i` ohne Ausgabe — es gibt keinen
    gebündelten ffprobe; die Stream-Zeilen im stderr reichen."""
    r = subprocess.run([get_ffmpeg_cli(), "-hide_banner", "-i", str(pfad)],
                       capture_output=True, text=True, errors="replace")
    video = audio = None
    for art, codec in _STREAM.findall(r.stderr):
        if art == "Video" and video is None:
            video = codec.lower()
        elif art == "Audio" and audio is None:
            audio = codec.lower()
    m = _MASSE.search(r.stderr)
    d = _DAUER.search(r.stderr)
    return {"video_codec": video, "audio_codec": audio,
            "breite": int(m.group(1)) if m else None,
            "hoehe": int(m.group(2)) if m else None,
            "dauer_s": (int(d.group(1)) * 3600 + int(d.group(2)) * 60
                        + float(d.group(3))) if d else None}


def pruefe(pfad: Path) -> dict | None:
    """None, wenn die Datei kein Video ist (reines Audio — geht seinen
    bisherigen Weg). Sonst die Sondierung, oder VideoFehler."""
    info = sondiere(pfad)
    if not info["video_codec"] or info["video_codec"] in ("mjpeg", "png"):
        return None            # Cover-Art in mp3/m4a zählt nicht als Video
    groesse = pfad.stat().st_size
    if groesse > VIDEO_MAX_BYTES:
        raise VideoFehler("zu-gross", (
            f"Video zu gross ({groesse / 1024 ** 3:.1f} GB, Grenze 10 GB)."))
    if pfad.suffix.lower() not in VIDEO_ENDUNGEN \
            or info["video_codec"] not in CODECS_OK:
        raise VideoFehler("codec", (
            f"Video nicht unterstützt ({info['video_codec']} in "
            f"{pfad.suffix.lower() or '?'}). LocalTranscript wandelt Video "
            "nicht um — bitte extern als H.264 oder HEVC in MP4 exportieren "
            "(z. B. HandBrake, QuickTime «Exportieren»)."))
    if not info["audio_codec"]:
        raise VideoFehler("ohne-ton", "Das Video hat keine Tonspur.")
    return info


def ton_extrahieren(quelle: Path, ziel: Path) -> Path:
    """Die Tonspur als mp3 (q2, wie der enrich-Export) — Arbeitskopie für
    Whisper, Editor und Exporte. Das Video selbst bleibt unangetastet."""
    r = subprocess.run(
        [get_ffmpeg_cli(), "-y", "-i", str(quelle), "-vn", "-c:a", "libmp3lame",
         "-q:a", "2", str(ziel)], capture_output=True)
    if r.returncode != 0 or not ziel.is_file():
        raise RuntimeError("Tonspur konnte nicht gelesen werden (ffmpeg)")
    return ziel
