"""LocalTranscript 2.0 — FastAPI-Backend (Loopback, Port 44100).

Die Shell (Tauri) ist dumm: sie spawnt uvicorn und lädt das UI; Config,
Bibliothek und Jobs besitzt das Backend. Statisches Frontend wird —
falls gebaut — same-origin mitserviert (Browser-Betrieb ohne App).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict

from . import bibliothek, config, exporte, jobs
from .bibliothek import BibliothekFehler

app = FastAPI(title=config.APP_NAME, version=config.APP_VERSION)

# Das Tauri-Fenster (Origin tauri://localhost) spricht 127.0.0.1:44100
# CROSS-origin — ohne CORS blockt WebKit die Antwort („Load failed",
# Live-Befund 2026-08-30). Allowlist statt "*": nur eigene Fenster;
# der Browser-Betrieb ist same-origin und braucht keins.
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["tauri://localhost", "http://tauri.localhost",
                   "http://localhost:1421"],
    allow_methods=["*"], allow_headers=["*"])

AUDIO_MEDIA = {".mp3": "audio/mpeg", ".m4a": "audio/mp4",
               ".aac": "audio/aac", ".wav": "audio/wav",
               ".ogg": "audio/ogg", ".flac": "audio/flac",
               ".webm": "audio/webm"}


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _err(e: Exception) -> HTTPException:
    if isinstance(e, BibliothekFehler):
        return HTTPException(status_code=404, detail=str(e))
    return HTTPException(status_code=500, detail=str(e))


# ---------- Health / Modelle / Einstellungen ----------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app": config.APP_NAME,
            "version": config.APP_VERSION}


@app.get("/api/models")
def models() -> dict:
    return {"models": config.get_available_models(),
            "models_dir": str(config.get_models_dir())}


@app.get("/api/settings")
def settings_get() -> dict:
    cfg = config.read_config()
    cfg["default_library_root"] = str(config.default_library_root())
    return cfg


class SettingsReq(ApiModel):
    model_config = ConfigDict(extra="allow")


@app.post("/api/settings")
def settings_post(req: SettingsReq) -> dict:
    aend = req.model_dump()
    root = aend.get("library_root")
    if root == "default":
        p = config.default_library_root()
        p.mkdir(parents=True, exist_ok=True)
        aend["library_root"] = str(p)
    elif root:
        p = Path(root).expanduser()
        if not p.is_dir():
            raise HTTPException(status_code=409,
                                detail=f"Kein Ordner: {p}")
        aend["library_root"] = str(p)
    return config.write_config(aend)


# ---------- Transkription ----------

def _params(model: str, language: str, speaker_range: str,
            cluster_threshold: float, diarize: bool) -> dict:
    min_sp = max_sp = 0
    if speaker_range and speaker_range != "auto" \
            and "-" in speaker_range:
        a, _, b = speaker_range.partition("-")
        try:
            min_sp, max_sp = int(a), int(b)
        except ValueError:
            min_sp = max_sp = 0
    return {"model": model, "language": language,
            "min_speakers": min_sp, "max_speakers": max_sp,
            "cluster_threshold": cluster_threshold, "diarize": diarize}


@app.post("/api/transcribe")
async def transcribe_upload(file: UploadFile = File(...),
                            model: str = Form("large-v3-turbo"),
                            language: str = Form("de"),
                            speaker_range: str = Form("auto"),
                            cluster_threshold: float = Form(0.5),
                            diarize: bool = Form(True)) -> dict:
    name = file.filename or "audio"
    endung = Path(name).suffix.lower()
    if endung not in bibliothek.AUDIO_ENDUNGEN:
        raise HTTPException(status_code=400,
                            detail=f"Nicht unterstützt: {endung}")
    tmp = Path(tempfile.mkstemp(suffix=endung, prefix="lt-up-")[1])
    with tmp.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    job = jobs.starte(tmp, name,
                      _params(model, language, speaker_range,
                              cluster_threshold, diarize),
                      quelle_ist_temp=True)
    return {"job_id": job["id"]}


class TranscribePathReq(ApiModel):
    path: str
    model: str = "large-v3-turbo"
    language: str = "de"
    speaker_range: str = "auto"
    cluster_threshold: float = 0.5
    diarize: bool = True


@app.post("/api/transcribe-path")
def transcribe_path(req: TranscribePathReq) -> dict:
    """App-Weg (nativer Datei-Drop): Pfad statt Multipart — große
    Audios werden nie durch HTTP kopiert."""
    p = Path(req.path).expanduser()
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"{p} fehlt")
    if p.suffix.lower() not in bibliothek.AUDIO_ENDUNGEN:
        raise HTTPException(status_code=400,
                            detail=f"Nicht unterstützt: {p.suffix}")
    job = jobs.starte(p, p.name,
                      _params(req.model, req.language,
                              req.speaker_range, req.cluster_threshold,
                              req.diarize))
    return {"job_id": job["id"]}


@app.get("/api/jobs")
def jobs_liste() -> dict:
    return {"jobs": [jobs.sicht(j) for j in jobs.JOBS.values()]}


@app.get("/api/jobs/{job_id}")
def job_get(job_id: str) -> dict:
    j = jobs.JOBS.get(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail="Job unbekannt")
    return jobs.sicht(j)


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    if job_id not in jobs.JOBS:
        raise HTTPException(status_code=404, detail="Job unbekannt")
    jobs.abbrechen(job_id)
    return {"status": "cancelling"}


# ---------- Import ----------

def _import_turns(turns: list[dict], name: str, quelle_art: str,
                  audio: Path | None) -> dict:
    if not turns:
        raise HTTPException(status_code=400,
                            detail="Keine Segmente gefunden")
    namen: list[str] = []
    for t in turns:
        sp = (t.get("speaker") or "").strip()
        if sp and sp not in namen:
            namen.append(sp)
    sprecher = [{"id": f"sp{i + 1}", "name": n}
                for i, n in enumerate(namen)]
    sid = {s["name"]: s["id"] for s in sprecher}
    segmente = [{"start": t["t0_s"], "end": t["t1_s"],
                 "sprecher": sid.get((t.get("speaker") or "").strip()),
                 "text": t["text"]} for t in turns]
    eintrag = bibliothek.anlegen(
        name=name, segmente=segmente, sprecher=sprecher,
        quelle={"datei": name, "erzeugt": quelle_art}, audio=audio)
    return {"eintrag": eintrag["id"], "segmente": len(segmente),
            "sprecher": len(sprecher)}


def _parse_import(daten: bytes, endung: str) -> list[dict]:
    from .enrich_export.turns import parse_transkript_csv, parse_transkript_vtt
    if endung in (".vtt", ".webvtt"):
        return parse_transkript_vtt(daten)
    if endung == ".csv":
        return parse_transkript_csv(daten)
    raise HTTPException(status_code=400,
                        detail=f"Nur .vtt/.csv — nicht {endung}")


@app.post("/api/import")
async def import_upload(datei: UploadFile = File(...),
                        audio: UploadFile | None = File(None)) -> dict:
    name = Path(datei.filename or "transkript")
    turns = _parse_import(await datei.read(), name.suffix.lower())
    audio_tmp: Path | None = None
    try:
        if audio is not None and audio.filename:
            endung = Path(audio.filename).suffix.lower()
            if endung in bibliothek.AUDIO_ENDUNGEN:
                audio_tmp = Path(tempfile.mkstemp(suffix=endung,
                                                  prefix="lt-imp-")[1])
                audio_tmp.write_bytes(await audio.read())
        return _import_turns(turns, name.stem,
                             f"import-{name.suffix.lstrip('.')}",
                             audio_tmp)
    finally:
        if audio_tmp is not None:
            audio_tmp.unlink(missing_ok=True)


class ImportPathReq(ApiModel):
    path: str
    audio_path: str | None = None


@app.post("/api/import-path")
def import_path(req: ImportPathReq) -> dict:
    p = Path(req.path).expanduser()
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"{p} fehlt")
    turns = _parse_import(p.read_bytes(), p.suffix.lower())
    audio: Path | None = None
    if req.audio_path:
        a = Path(req.audio_path).expanduser()
        if a.is_file():
            audio = a
    if audio is None:
        for endung in bibliothek.AUDIO_ENDUNGEN:
            k = p.with_suffix(endung)
            if k.is_file():
                audio = k
                break
    return _import_turns(turns, p.stem,
                         f"import-{p.suffix.lstrip('.')}", audio)


# ---------- Bibliothek ----------

@app.get("/api/transcripts")
def transcripts() -> dict:
    return {"transcripts": bibliothek.liste(),
            "library_root": str(config.library_root() or "")}


@app.get("/api/transcripts/{eid}")
def transcript_get(eid: str) -> dict:
    try:
        return bibliothek.lese(eid)
    except BibliothekFehler as e:
        raise _err(e) from e


class SaveReq(ApiModel):
    sprecher: list[dict]
    segmente: list[dict]


@app.put("/api/transcripts/{eid}")
def transcript_put(eid: str, req: SaveReq) -> dict:
    try:
        d = bibliothek.lese(eid)
    except BibliothekFehler as e:
        raise _err(e) from e
    ids = {s.get("id") for s in req.sprecher}
    for seg in req.segmente:
        if seg.get("sprecher") and seg["sprecher"] not in ids:
            raise HTTPException(status_code=422,
                                detail=f"Unbekannter Sprecher: "
                                       f"{seg['sprecher']}")
    d["sprecher"] = req.sprecher
    d["segmente"] = req.segmente
    d = bibliothek.schreibe(eid, d)
    return {"status": "saved", "updated": d["updated"]}


class RenameReq(ApiModel):
    name: str


@app.post("/api/transcripts/{eid}/rename")
def transcript_rename(eid: str, req: RenameReq) -> dict:
    try:
        d = bibliothek.umbenennen(eid, req.name)
    except BibliothekFehler as e:
        raise _err(e) from e
    return {"status": "renamed", "name": d["name"]}


class DeleteReq(ApiModel):
    confirm: str


@app.post("/api/transcripts/{eid}/delete")
def transcript_delete(eid: str, req: DeleteReq) -> dict:
    if req.confirm != eid:
        raise HTTPException(status_code=409,
                            detail="confirm muss die ID sein")
    try:
        bibliothek.loeschen(eid)
    except BibliothekFehler as e:
        raise _err(e) from e
    return {"status": "deleted"}


@app.get("/api/transcripts/{eid}/audio")
def transcript_audio(eid: str):
    try:
        p = bibliothek.audio_pfad(eid)
    except BibliothekFehler as e:
        raise _err(e) from e
    if p is None:
        raise HTTPException(status_code=404, detail="Kein Audio")
    return FileResponse(p, media_type=AUDIO_MEDIA.get(
        p.suffix.lower(), "application/octet-stream"))


@app.get("/api/transcripts/{eid}/sprecher/{sid}/sample")
def sprecher_sample(eid: str, sid: str, tasks: BackgroundTasks):
    """~8-s-Hörprobe: das LÄNGSTE Segment dieses Sprechers, aus dem
    Bibliotheks-Audio geschnitten (v2: braucht keine Diarisierungs-
    Rohdaten mehr — funktioniert auch für Importe und nach Neustarts)."""
    import subprocess

    from .config import get_ffmpeg_cli
    try:
        d = bibliothek.lese(eid)
        audio = bibliothek.audio_pfad(eid)
    except BibliothekFehler as e:
        raise _err(e) from e
    if audio is None:
        raise HTTPException(status_code=404, detail="Kein Audio")
    kandidaten = [s for s in d["segmente"] if s.get("sprecher") == sid]
    if not kandidaten:
        raise HTTPException(status_code=404, detail="Sprecher leer")
    seg = max(kandidaten, key=lambda s: s["end"] - s["start"])
    dauer = min(8.0, max(seg["end"] - seg["start"], 1.0))
    tmp = Path(tempfile.mkstemp(suffix=".wav", prefix="lt-sample-")[1])
    r = subprocess.run(
        [get_ffmpeg_cli(), "-y", "-ss", f"{seg['start']:.3f}",
         "-t", f"{dauer:.3f}", "-i", str(audio), "-ar", "16000",
         "-ac", "1", "-c:a", "pcm_s16le", str(tmp)],
        capture_output=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500,
                            detail="Probe fehlgeschlagen (ffmpeg)")
    tasks.add_task(tmp.unlink, True)
    return FileResponse(tmp, media_type="audio/wav")


# ---------- Export ----------

@app.get("/api/transcripts/{eid}/export/{format}")
def export_download(eid: str, format: str) -> Response:
    try:
        inhalt, name, media = exporte.export_bytes(eid, format)
    except (BibliothekFehler, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return Response(inhalt, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="{name}"'})


class ExportReq(ApiModel):
    format: str
    path: str


@app.post("/api/transcripts/{eid}/export")
def export_datei(eid: str, req: ExportReq) -> dict:
    """App-Weg: nativer Save-Dialog liefert den Zielpfad."""
    try:
        inhalt, _name, _media = exporte.export_bytes(eid, req.format)
    except (BibliothekFehler, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    ziel = Path(req.path).expanduser()
    if not ziel.parent.is_dir():
        raise HTTPException(status_code=409,
                            detail=f"Ordner fehlt: {ziel.parent}")
    ziel.write_bytes(inhalt)
    return {"status": "exported", "path": str(ziel)}


# ---------- Statisches Frontend (Browser-Betrieb) ----------

_DIST = Path(__file__).resolve().parent.parent.parent.parent \
    / "frontend" / "dist"
# Im Bundle liegt das GEBAUTE Frontend unter Resources/frontend; im
# Dev-Repo ist frontend/ der QUELL-Ordner (eigene index.html!) — dort
# zählt nur dist/.
_KANDIDATEN = ([config.get_app_root() / "frontend"]
               if config.is_bundled() else []) + [_DIST]
for kandidat in _KANDIDATEN:
    if (kandidat / "index.html").is_file():
        import mimetypes

        from fastapi.staticfiles import StaticFiles

        # macOS-Python kennt .js teils nur als octet-stream —
        # Strict-MIME-Checking bricht dann jedes ES-Modul
        mimetypes.add_type("text/javascript", ".js")
        mimetypes.add_type("text/css", ".css")
        app.mount("/", StaticFiles(directory=kandidat, html=True),
                  name="static")
        break


def cli() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=config.PORT)
