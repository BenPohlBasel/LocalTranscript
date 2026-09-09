"""Transkriptions-Jobs: in-memory (bewusst flüchtig — das ERGEBNIS
landet sofort als Bibliotheks-Eintrag, v1 verlor bei jedem Neustart
alle Jobs UND ihre Bearbeitbarkeit).

Pipeline je Job (eigener Thread):
  1. ffmpeg → 16 kHz mono WAV (IMMER — v1 ließ .wav-Uploads unkonvertiert
     zu whisper.cpp durch, ein gemessener v1-Bug)
  2. optional Diarisierung (silero-vad + ECAPA + Clustering)
  3. whisper.cpp: je Sprecher-Block ein Clip (diarisiert) oder die
     ganze Datei mit Live-Text-Streaming (klassisch)
  4. Bibliotheks-Eintrag anlegen (Sprecher als Entitäten)

Abbruch KOOPERATIV UND HART: cancel-Event wird zwischen den Schritten
geprüft, ein laufender whisper-/ffmpeg-Prozess wird gekillt (v1s
DELETE ließ whisper einfach weiterrechnen).
"""
from __future__ import annotations

import subprocess
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from . import bibliothek, config
from .config import get_ffmpeg_cli, read_config
from .transcribe import (
    transcribe_classic,
    transcribe_segment,
)

JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


class Abbruch(Exception):
    pass


def _neu(filename: str, params: dict) -> dict:
    job = {"id": uuid.uuid4().hex[:8], "filename": filename,
           "params": params, "status": "pending", "progress": 0,
           "message": "", "partial_text": "", "error": None,
           "eintrag": None,
           "created_at": datetime.now(UTC).isoformat(
               timespec="seconds"),
           # gesetzt, wenn die Arbeit WIRKLICH beginnt — bei mehreren
           # gedroppten Dateien liegt created_at weit davor, und die
           # Anzeige „vergangen" wäre gelogen (User 2026-09-09)
           "started_at": None}
    with _LOCK:
        JOBS[job["id"]] = job
    return job


def _setze(job: dict, **kv) -> None:
    job.update(kv)


_CANCEL: dict[str, threading.Event] = {}

# ---------- Warteschlange ----------
# Vorher startete jeder Job sofort seinen Thread — vier gedroppte
# Dateien rechneten gleichzeitig um dieselbe GPU. Jetzt wartet der
# Thread ganz am Anfang auf einen Platz; der Job bleibt so lange
# „pending" mit started_at=None, und die Oberfläche zeigt „wartet …".
# Streng der Reihe nach (FIFO), damit die zuerst gedroppte Datei auch
# zuerst drankommt.
_SLOT = threading.Condition()
_LAUFEND: set[str] = set()
_WARTEND: list[str] = []


def _nimm_slot(job: dict) -> None:
    jid = job["id"]
    with _SLOT:
        _WARTEND.append(jid)
        while True:
            if _CANCEL.get(jid, threading.Event()).is_set():
                _WARTEND.remove(jid)
                raise Abbruch
            if _WARTEND[0] == jid and len(_LAUFEND) < config.max_parallel():
                _WARTEND.pop(0)
                _LAUFEND.add(jid)
                return
            _SLOT.wait(0.5)


def _gib_slot(job: dict) -> None:
    with _SLOT:
        _LAUFEND.discard(job["id"])
        if job["id"] in _WARTEND:
            _WARTEND.remove(job["id"])
        _SLOT.notify_all()
_PROC: dict[str, subprocess.Popen | None] = {}


def _pruefe_abbruch(job: dict) -> None:
    ev = _CANCEL.get(job["id"])
    if ev is not None and ev.is_set():
        raise Abbruch()


def _register_fuer(job_id: str):
    def reg(proc: subprocess.Popen | None) -> None:
        _PROC[job_id] = proc
    return reg


def abbrechen(job_id: str) -> bool:
    ev = _CANCEL.get(job_id)
    if ev is None:
        return False
    ev.set()
    proc = _PROC.get(job_id)
    if proc is not None and proc.poll() is None:
        proc.kill()
    return True


def merge_consecutive_speakers(diarization: list,
                               min_segment_duration: float = 0.5) -> list:
    """v1-Logik unverändert: Mikro-Segmente (<0.5 s) dem Vorgänger
    zuschlagen, dann gleiche aufeinanderfolgende Sprecher verschmelzen."""
    if not diarization:
        return []
    from .diarize import SpeakerSegment

    filtered: list = []
    for seg in sorted(diarization, key=lambda x: x.start):
        if seg.end - seg.start < min_segment_duration:
            if filtered:
                filtered[-1] = SpeakerSegment(
                    start=filtered[-1].start, end=seg.end,
                    speaker=filtered[-1].speaker)
            continue
        filtered.append(seg)
    merged: list = []
    for seg in filtered:
        if merged and merged[-1].speaker == seg.speaker:
            merged[-1] = SpeakerSegment(start=merged[-1].start,
                                        end=seg.end, speaker=seg.speaker)
        else:
            merged.append(seg)
    return merged


def _konvertiere(job: dict, quelle: Path, arbeits_dir: Path) -> Path:
    """IMMER nach 16 kHz mono WAV (auch .wav-Quellen — Resample-Fix)."""
    ziel = arbeits_dir / f"{job['id']}.wav"
    _nimm_slot(job)
    _setze(job, started_at=datetime.now(UTC).isoformat(
        timespec="seconds"), progress=5, message="konvertiere")
    proc = subprocess.Popen(
        [get_ffmpeg_cli(), "-y", "-i", str(quelle), "-ar", "16000",
         "-ac", "1", "-c:a", "pcm_s16le", str(ziel)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _PROC[job["id"]] = proc
    proc.wait()
    _PROC[job["id"]] = None
    _pruefe_abbruch(job)
    if proc.returncode != 0 or not ziel.is_file():
        raise RuntimeError("Audio-Konvertierung fehlgeschlagen (ffmpeg)")
    return ziel


def _clip(job: dict, wav: Path, start: float, end: float,
          i: int) -> Path:
    ziel = wav.parent / f"{job['id']}_seg{i}.wav"
    proc = subprocess.Popen(
        [get_ffmpeg_cli(), "-y", "-i", str(wav), "-ss", f"{start:.3f}",
         "-to", f"{end:.3f}", "-ar", "16000", "-ac", "1", "-c:a",
         "pcm_s16le", str(ziel)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _PROC[job["id"]] = proc
    proc.wait()
    _PROC[job["id"]] = None
    _pruefe_abbruch(job)
    return ziel


def _sprecher_label(n: int) -> str:
    basis = {"de": "Sprecher", "en": "Speaker", "fr": "Locuteur",
             "it": "Parlante"}.get(
        read_config().get("ui_language", "de"), "Speaker")
    return f"{basis} {n}"


def _lauf(job: dict, quelle: Path, name: str) -> None:
    import shutil
    import tempfile

    p = job["params"]
    tmp = Path(tempfile.mkdtemp(prefix="lt-job-"))
    try:
        wav = _konvertiere(job, quelle, tmp)
        segmente: list[dict] = []
        sprecher: list[dict] = []

        if p["diarize"]:
            _setze(job, status="diarizing", progress=10,
                   message="sprecher")
            from .diarize import diarize_audio
            # 10 → 25 % füllen, sonst steht der Balken die ganze
            # Diarisierung still; nebenbei greift der Abbruch dann
            # SOFORT und nicht erst nach dem ganzen Lauf.
            def _diar_fortschritt(i: int, n: int) -> None:
                _pruefe_abbruch(job)
                _setze(job, progress=10 + int(i / max(n, 1) * 15))

            diar = diarize_audio(str(wav), p["min_speakers"],
                                 p["max_speakers"],
                                 p["cluster_threshold"],
                                 fortschritt=_diar_fortschritt)
            _pruefe_abbruch(job)
            bloecke = merge_consecutive_speakers(diar)
            labels: dict[str, str] = {}   # SPEAKER_00 → Entitäts-ID
            # Reihenfolge nach Gesamt-Sprechzeit (dominanter zuerst)
            zeit: dict[str, float] = {}
            for b in bloecke:
                zeit[b.speaker] = zeit.get(b.speaker, 0.0) \
                    + (b.end - b.start)
            for n, roh in enumerate(sorted(zeit, key=zeit.get,
                                           reverse=True), 1):
                sid = f"sp{n}"
                labels[roh] = sid
                sprecher.append({"id": sid,
                                 "name": _sprecher_label(n)})
            _setze(job, status="transcribing",
                   message=f"transkribiere:0/{len(bloecke)}")
            reg = _register_fuer(job["id"])
            texte: list[str] = []
            for i, b in enumerate(bloecke):
                _pruefe_abbruch(job)
                _setze(job, progress=25 + int(i / max(len(bloecke), 1)
                                              * 60),
                       message=f"transkribiere:{i + 1}/{len(bloecke)}")
                clip = _clip(job, wav, b.start, b.end, i)
                try:
                    teile = transcribe_segment(
                        clip, p["model"], p["language"],
                        time_offset=b.start, register=reg)
                finally:
                    clip.unlink(missing_ok=True)
                _pruefe_abbruch(job)
                for t in teile:
                    segmente.append({"start": round(t.start, 3),
                                     "end": round(t.end, 3),
                                     "sprecher": labels[b.speaker],
                                     "text": t.text})
                    texte.append(t.text)
                _setze(job, partial_text=" ".join(texte[-80:]))
        else:
            _setze(job, status="transcribing", progress=15,
                   message="transkribiere")

            def cb(pct: int, _msg: str, text: str) -> None:
                _pruefe_abbruch(job)
                if pct >= 0:
                    _setze(job, progress=15 + int(pct * 0.7))
                _setze(job, partial_text=text[-4000:])

            teile = transcribe_classic(wav, p["model"], p["language"],
                                       progress_cb=cb,
                                       register=_register_fuer(job["id"]))
            _pruefe_abbruch(job)
            segmente = [{"start": round(t.start, 3),
                         "end": round(t.end, 3), "sprecher": None,
                         "text": t.text} for t in teile]

        _setze(job, status="saving", progress=90, message="speichere")
        eintrag = bibliothek.anlegen(
            name=name, segmente=segmente, sprecher=sprecher,
            quelle={"datei": job["filename"], "model": p["model"],
                    "language": p["language"], "diarize": p["diarize"],
                    "erzeugt": "transcription"},
            audio=quelle)
        _setze(job, status="completed", progress=100, message="fertig",
               eintrag=eintrag["id"])
    except Abbruch:
        _setze(job, status="cancelled", message="abgebrochen")
    except Exception as e:  # noqa: BLE001 — Job-Fehler gehören in den
        # Job-Status, nie als toter Thread verschluckt
        _setze(job, status="failed", error=str(e), message="fehler")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        quelle_tmp = job.get("_quelle_tmp")
        if quelle_tmp:
            Path(quelle_tmp).unlink(missing_ok=True)
        _gib_slot(job)
        _CANCEL.pop(job["id"], None)
        _PROC.pop(job["id"], None)


def starte(quelle: Path, filename: str, params: dict,
           quelle_ist_temp: bool = False) -> dict:
    job = _neu(filename, params)
    if quelle_ist_temp:
        job["_quelle_tmp"] = str(quelle)
    _CANCEL[job["id"]] = threading.Event()
    t = threading.Thread(target=_lauf, name=f"lt-job-{job['id']}",
                         args=(job, quelle, Path(filename).stem),
                         daemon=True)
    t.start()
    return job


def sicht(job: dict) -> dict:
    return {k: v for k, v in job.items() if not k.startswith("_")}
