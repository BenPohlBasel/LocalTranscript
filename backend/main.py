"""
LocalTranscript - FastAPI Backend
Audio transcription with speaker diarization

Entwickelt von BIAS.City - MIT Lizenz
Kontakt: kontakt@BIAS.City
"""

import os
import uuid
import asyncio
import socket
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv, set_key

from .config import get_ffmpeg_cli, get_models_dir, UPLOADS_DIR, OUTPUTS_DIR, APP_ROOT

# Version info
APP_VERSION = "1.0.0"
APP_NAME = "LocalTranscript"

# Project paths
BASE_DIR = Path(__file__).parent.parent
ENV_FILE = BASE_DIR / ".env"


def _get_ffmpeg():
    """Get ffmpeg path"""
    return get_ffmpeg_cli()


def _get_models_dir():
    """Get models directory"""
    return get_models_dir()


def _get_local_ip():
    """Get local IP address for LAN access"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """Delete files older than max_age_hours"""
    if not directory.exists():
        return 0

    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    deleted = 0

    for f in directory.iterdir():
        if f.is_file():
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                try:
                    f.unlink()
                    deleted += 1
                except Exception:
                    pass

    return deleted


# Load environment
load_dotenv(ENV_FILE)

app = FastAPI(title=APP_NAME, version=APP_VERSION)


@app.on_event("startup")
async def startup_cleanup():
    """Clean old uploads and outputs on startup"""
    uploads_deleted = cleanup_old_files(UPLOADS_DIR, max_age_hours=24)
    outputs_deleted = cleanup_old_files(OUTPUTS_DIR, max_age_hours=48)
    if uploads_deleted or outputs_deleted:
        print(f"Cleanup: {uploads_deleted} uploads, {outputs_deleted} outputs gelöscht")

# CORS for LAN access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    MERGING = "merging"
    COMPLETED = "completed"
    FAILED = "failed"


# In-memory job storage
jobs: dict[str, dict] = {}


def get_available_models() -> list[dict]:
    """Scan models directory for available GGML models"""
    models = []
    models_dir = _get_models_dir()
    if models_dir.exists():
        for f in models_dir.glob("ggml-*.bin"):
            name = f.stem.replace("ggml-", "")
            size_mb = f.stat().st_size / (1024 * 1024)
            models.append({
                "name": name,
                "path": str(f),
                "size_mb": round(size_mb, 1)
            })
    return sorted(models, key=lambda x: x["size_mb"])


def has_hf_token() -> bool:
    """Check if HuggingFace token is configured"""
    token = os.environ.get("HF_TOKEN", "")
    return bool(token and token != "DEIN_TOKEN_HIER" and len(token) > 10)


class TokenRequest(BaseModel):
    token: str


@app.get("/api/health")
async def healthcheck():
    """Lightweight readiness probe (used by the Electron launcher)"""
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/setup")
async def get_setup_status():
    """Check if setup is complete"""
    return {
        "has_token": has_hf_token(),
        "models_available": len(get_available_models()) > 0
    }


@app.post("/api/setup/token")
async def save_token(request: TokenRequest):
    """Save HuggingFace token to .env file"""
    token = request.token.strip()

    if not token.startswith("hf_"):
        raise HTTPException(
            status_code=400,
            detail="Token muss mit 'hf_' beginnen"
        )

    # Create .env if not exists
    if not ENV_FILE.exists():
        ENV_FILE.touch()

    # Save token
    set_key(str(ENV_FILE), "HF_TOKEN", token)

    # Update environment
    os.environ["HF_TOKEN"] = token

    # Reload diarization module
    try:
        import diarize
        diarize.HF_TOKEN = token
        diarize._pipeline = None  # Reset pipeline to reload with new token
    except:
        pass

    return {"status": "saved", "message": "Token gespeichert!"}


@app.get("/api/models")
async def list_models():
    """List available Whisper models"""
    return {"models": get_available_models()}


@app.get("/api/info")
async def get_info(request: Request):
    """Get system info, API endpoints, licenses, and DSGVO information"""
    models = get_available_models()
    has_token = bool(os.environ.get("HF_TOKEN"))

    # Use the URL the client actually connected to so we report the correct
    # (dynamic) port instead of a hardcoded one.
    base = str(request.base_url).rstrip('/')
    api_local = f"{base}/api"

    return {
        "app": {
            "name": APP_NAME,
            "version": APP_VERSION,
            "developer": "BIAS.City",
            "contact": "kontakt@BIAS.City",
            "license": "MIT",
            "repository": "https://github.com/BenPohlBasel/LocalTranscript"
        },
        "api": {
            "local": api_local,
            "endpoints": {
                "GET /api/health": "Readiness-Probe",
                "POST /api/transcribe": "Audio hochladen und transkribieren",
                "GET /api/jobs": "Alle Jobs auflisten",
                "GET /api/jobs/{id}": "Job-Status abfragen",
                "GET /api/jobs/{id}/vtt": "VTT-Datei herunterladen",
                "GET /api/jobs/{id}/csv": "CSV-Datei herunterladen",
                "GET /api/jobs/{id}/txt": "Plain-Text-Datei herunterladen",
                "POST /api/jobs/{id}/save-vtt": "VTT in Downloads-Ordner speichern",
                "POST /api/jobs/{id}/save-csv": "CSV in Downloads-Ordner speichern",
                "POST /api/jobs/{id}/save-txt": "TXT in Downloads-Ordner speichern",
                "DELETE /api/jobs/{id}": "Job abbrechen / löschen",
                "GET /api/models": "Verfügbare Whisper-Modelle",
                "GET /api/info": "Diese Informationen"
            }
        },
        "technology": {
            "transcription": {
                "name": "Whisper",
                "implementation": "whisper.cpp (C++ mit Metal GPU)",
                "models": [m["name"] for m in models],
                "license": "MIT (OpenAI)"
            },
            "diarization": {
                "name": "SpeechBrain ECAPA-TDNN + silero-vad",
                "model": "spkrec-ecapa-voxceleb",
                "available": True,
                "license": "Apache 2.0 (SpeechBrain) + MIT (silero)"
            },
            "backend": {
                "framework": "FastAPI + Uvicorn",
                "language": "Python 3.11+",
                "license": "MIT"
            },
            "frontend": {
                "type": "HTML/CSS/JavaScript",
                "native_wrapper": "PyWebView",
                "license": "MIT"
            }
        },
        "privacy": {
            "data_location": "Alle Daten bleiben lokal auf diesem Gerät",
            "cloud_connection": "Keine - vollständig offline nach Installation",
            "uploads": f"Gespeichert in: {UPLOADS_DIR}",
            "auto_cleanup": "Uploads werden nach 24h automatisch gelöscht",
            "dsgvo": {
                "compliant": True,
                "reason": "Keine Datenübertragung an Dritte, alle Verarbeitung lokal",
                "data_subject_rights": "Volle Kontrolle über alle Daten auf eigenem Gerät"
            }
        },
        "system": {
            "requirements": {
                "processor": "Apple Silicon (M1/M2/M3/M4) erforderlich",
                "ram": "16 GB empfohlen (8 GB minimum)",
                "storage": "~10 GB (Modelle + Cache)",
                "macos": "11.0 (Big Sur) oder neuer"
            }
        }
    }


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs"""
    return {"jobs": list(jobs.values())}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.post("/api/transcribe")
async def create_transcription(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form("medium"),
    language: str = Form("de"),
    speaker_range: str = Form("auto"),
    cluster_threshold: float = Form(0.5),
    diarize: bool = Form(True)
):
    """Upload audio file and start transcription"""

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    # Check model exists
    available = [m["name"] for m in get_available_models()]
    if model not in available:
        raise HTTPException(status_code=400, detail=f"Model not found. Available: {available}")

    # Create job
    job_id = str(uuid.uuid4())[:8]
    upload_path = UPLOADS_DIR / f"{job_id}{ext}"
    output_path = OUTPUTS_DIR / f"{job_id}.vtt"

    # Save uploaded file
    content = await file.read()
    upload_path.write_bytes(content)

    # Parse speaker range
    min_speakers = 0
    max_speakers = 0
    if speaker_range != "auto":
        parts = speaker_range.split("-")
        if len(parts) == 2:
            min_speakers = int(parts[0])
            max_speakers = int(parts[1])

    # Create job record
    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "model": model,
        "language": language,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "cluster_threshold": cluster_threshold,
        "diarize": diarize,
        "status": JobStatus.PENDING,
        "progress": 0,
        "message": "Waiting to start...",
        "created_at": datetime.now().isoformat(),
        "upload_path": str(upload_path),
        "output_path": str(output_path),
        "error": None,
        "partial_text": ""
    }

    # Start processing in background
    background_tasks.add_task(process_job, job_id)

    return {"job_id": job_id, "status": "created"}


async def process_job(job_id: str):
    """Process a transcription job"""
    from .diarize import diarize_audio
    from .transcribe import transcribe_classic, transcribe_segment_with_timestamps
    from .merge import (
        build_vtt_from_speaker_transcript_segments,
        build_csv_from_speaker_transcript_segments,
        build_txt_from_speaker_transcript_segments,
        build_vtt_from_transcript_segments_no_speakers,
        build_csv_from_transcript_segments_no_speakers,
        build_txt_from_transcript_segments_no_speakers,
        format_speaker_name,
    )

    job = jobs[job_id]
    temp_files = []  # Track temp files for cleanup

    def update_progress(percent: int, message: str, partial_text: str = ""):
        """Callback for real-time progress updates"""
        if percent >= 0:
            job["progress"] = percent
            job["message"] = message
        if partial_text:
            job["partial_text"] = partial_text

    upload_path = Path(job["upload_path"])
    wav_path = upload_path.with_suffix(".wav")

    try:
        # Convert to 16kHz WAV if needed
        if upload_path.suffix.lower() != ".wav":
            job["progress"] = 5
            job["message"] = "Konvertiere Audio..."
            import subprocess
            cmd = [
                _get_ffmpeg(), "-y",
                "-i", str(upload_path),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(wav_path)
            ]
            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Audio conversion failed: {result.stderr}")
            audio_path = str(wav_path)
        else:
            audio_path = str(upload_path)

        # No-diarize path: transcribe whole file once, skip speaker logic
        if not job.get("diarize", True):
            job["status"] = JobStatus.TRANSCRIBING
            job["progress"] = 15
            job["message"] = "Transkribiere Audio..."

            def _progress(percent: int, message: str, partial_text: str = ""):
                # transcribe_classic maps whisper 0-100 to overall 35-80.
                # Re-map to 15-85 so the bar moves steadily in this mode.
                if percent >= 35:
                    whisper_pct = (percent - 35) / 45 * 100  # back to 0-100
                    overall = 15 + int(whisper_pct * 0.70)
                    job["progress"] = max(15, min(85, overall))
                if message:
                    job["message"] = message
                if partial_text:
                    job["partial_text"] = partial_text

            transcript_segments = await asyncio.to_thread(
                transcribe_classic,
                audio_path,
                job["model"],
                job["language"],
                _progress,
            )

            if transcript_segments:
                job["partial_text"] = " ".join(s.text for s in transcript_segments)

            job["status"] = JobStatus.MERGING
            job["progress"] = 90
            job["message"] = "Erstelle VTT..."

            vtt_content = build_vtt_from_transcript_segments_no_speakers(transcript_segments)
            csv_content = build_csv_from_transcript_segments_no_speakers(transcript_segments)
            txt_content = build_txt_from_transcript_segments_no_speakers(transcript_segments)

            Path(job["output_path"]).write_text(vtt_content, encoding="utf-8")
            csv_path = Path(job["output_path"]).with_suffix(".csv")
            csv_path.write_text(csv_content, encoding="utf-8")
            job["csv_path"] = str(csv_path)
            txt_path = Path(job["output_path"]).with_suffix(".txt")
            txt_path.write_text(txt_content, encoding="utf-8")
            job["txt_path"] = str(txt_path)

            job["status"] = JobStatus.COMPLETED
            job["progress"] = 100
            job["message"] = "Transcription complete!"
            return

        # Step 1: Diarize - get speaker segments
        job["status"] = JobStatus.DIARIZING
        job["progress"] = 10
        job["message"] = "Erkenne Sprecher..."

        diarization = await asyncio.to_thread(
            diarize_audio,
            audio_path,
            min_speakers=job.get("min_speakers", 0),
            max_speakers=job.get("max_speakers", 0),
            threshold=job.get("cluster_threshold", 0.5)
        )

        # Merge consecutive segments from same speaker
        merged_diar = merge_consecutive_speakers(diarization)
        num_speakers = len(set(s.speaker for s in diarization))
        job["progress"] = 25
        job["message"] = f"{num_speakers} Sprecher erkannt. Transkribiere {len(merged_diar)} Segmente..."
        print(f"Diarization: {num_speakers} speakers, {len(merged_diar)} merged segments")

        # Persist the diarization (after merging) so the speaker-sample and
        # rename endpoints can work without a re-run. Stored as plain dicts so
        # the job dict stays JSON-serialisable.
        job["diarization"] = [
            {"start": s.start, "end": s.end, "speaker": s.speaker}
            for s in merged_diar
        ]
        # Speakers ordered by total speech time (matches diarize.py post-processing)
        speaker_durations: dict[str, float] = {}
        for s in merged_diar:
            speaker_durations[s.speaker] = speaker_durations.get(s.speaker, 0.0) + (s.end - s.start)
        job["speakers"] = sorted(speaker_durations, key=lambda sp: -speaker_durations[sp])
        job["speaker_names"] = {sp: format_speaker_name(sp) for sp in job["speakers"]}

        # Step 2: Per-speaker-block transcription. We cut the audio at the
        # diarization boundaries and run whisper-cli on each clip, so each
        # whisper call contains exactly one speaker — no post-hoc word-to-
        # speaker matching needed. whisper.cpp Metal makes the per-call
        # overhead negligible.
        job["status"] = JobStatus.TRANSCRIBING
        all_segments = []
        all_text_parts = []

        for i, diar_seg in enumerate(merged_diar):
            progress = 25 + int((i / len(merged_diar)) * 55)
            speaker_name = format_speaker_name(diar_seg.speaker)
            job["progress"] = progress
            job["message"] = f"Transkribiere Segment {i + 1}/{len(merged_diar)} ({speaker_name})..."

            segment_path = upload_path.parent / f"{job_id}_seg{i}.wav"
            temp_files.append(segment_path)

            import subprocess
            extract_cmd = [
                _get_ffmpeg(), "-y",
                "-i", audio_path,
                "-ss", str(diar_seg.start),
                "-to", str(diar_seg.end),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(segment_path),
            ]
            await asyncio.to_thread(subprocess.run, extract_cmd, capture_output=True)

            # Transcribe this single-speaker clip with absolute timestamps
            transcript_segments = await asyncio.to_thread(
                transcribe_segment_with_timestamps,
                str(segment_path),
                job["model"],
                job["language"],
                diar_seg.start,
            )

            if transcript_segments:
                all_segments.append({
                    "speaker": diar_seg.speaker,
                    "segments": transcript_segments,
                })
                for seg in transcript_segments:
                    all_text_parts.append(seg.text)
                job["partial_text"] = " ".join(all_text_parts)

        job["progress"] = 85

        # Step 3: Build VTT and CSV
        job["status"] = JobStatus.MERGING
        job["progress"] = 90
        job["message"] = "Erstelle VTT..."

        vtt_content = build_vtt_from_speaker_transcript_segments(all_segments)
        csv_content = build_csv_from_speaker_transcript_segments(all_segments)
        txt_content = build_txt_from_speaker_transcript_segments(all_segments)

        # Save VTT, CSV and TXT
        Path(job["output_path"]).write_text(vtt_content, encoding="utf-8")
        csv_path = Path(job["output_path"]).with_suffix(".csv")
        csv_path.write_text(csv_content, encoding="utf-8")
        job["csv_path"] = str(csv_path)
        txt_path = Path(job["output_path"]).with_suffix(".txt")
        txt_path.write_text(txt_content, encoding="utf-8")
        job["txt_path"] = str(txt_path)

        # Done
        job["status"] = JobStatus.COMPLETED
        job["progress"] = 100
        job["message"] = "Transcription complete!"

    except Exception as e:
        job["status"] = JobStatus.FAILED
        job["error"] = str(e)
        job["message"] = f"Error: {str(e)}"
        print(f"Job {job_id} failed: {e}")

    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
        # Cleanup converted WAV
        if wav_path.exists() and wav_path != upload_path:
            try:
                wav_path.unlink()
            except:
                pass


def merge_consecutive_speakers(diarization: list, min_segment_duration: float = 0.5) -> list:
    """
    Merge consecutive segments from the same speaker.
    Also filters out very short segments (< min_segment_duration) by assigning them
    to the surrounding speaker.
    """
    if not diarization:
        return []

    from .diarize import SpeakerSegment

    # Sort by start time
    sorted_diar = sorted(diarization, key=lambda x: x.start)

    # Step 1: Filter out micro-segments (< min_segment_duration)
    # Assign them to the previous or next speaker
    filtered = []
    for i, seg in enumerate(sorted_diar):
        duration = seg.end - seg.start

        if duration < min_segment_duration:
            # Very short segment - try to merge with neighbors
            if filtered:
                # Extend previous segment to cover this one
                filtered[-1] = SpeakerSegment(
                    start=filtered[-1].start,
                    end=seg.end,
                    speaker=filtered[-1].speaker
                )
            # If no previous segment, skip this micro-segment
            continue

        filtered.append(seg)

    # Step 2: Merge consecutive segments from same speaker
    merged = []
    for seg in filtered:
        if merged and merged[-1].speaker == seg.speaker:
            # Same speaker - extend the previous segment
            merged[-1] = SpeakerSegment(
                start=merged[-1].start,
                end=seg.end,
                speaker=seg.speaker
            )
        else:
            merged.append(seg)

    return merged


@app.get("/api/jobs/{job_id}/vtt")
async def download_vtt(job_id: str):
    """Download VTT file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    output_path = Path(job["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="VTT file not found")

    # Generate filename from original
    original_name = Path(job["filename"]).stem
    download_name = f"{original_name}_speakers.vtt"

    return FileResponse(
        output_path,
        media_type="text/vtt",
        filename=download_name
    )


@app.get("/api/jobs/{job_id}/csv")
async def download_csv(job_id: str):
    """Download CSV file with speaker paragraphs"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    csv_path = Path(job.get("csv_path", ""))
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")

    # Generate filename from original
    original_name = Path(job["filename"]).stem
    download_name = f"{original_name}_speakers.csv"

    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=download_name
    )


@app.get("/api/jobs/{job_id}/txt")
async def download_txt(job_id: str):
    """Download plain-text file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    txt_path = Path(job.get("txt_path", ""))
    if not txt_path.exists():
        raise HTTPException(status_code=404, detail="TXT file not found")

    original_name = Path(job["filename"]).stem
    download_name = f"{original_name}_transkript.txt"

    return FileResponse(
        txt_path,
        media_type="text/plain",
        filename=download_name
    )


@app.post("/api/jobs/{job_id}/save-vtt")
async def save_vtt_to_downloads(job_id: str):
    """Save VTT file to user's Downloads folder"""
    import shutil

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    vtt_path = Path(job.get("output_path", ""))
    if not vtt_path.exists():
        raise HTTPException(status_code=404, detail="VTT file not found")

    # Save to Downloads folder
    downloads_dir = Path.home() / "Downloads"
    original_name = Path(job["filename"]).stem
    dest_path = downloads_dir / f"{original_name}_transkript.vtt"

    # Avoid overwriting
    counter = 1
    while dest_path.exists():
        dest_path = downloads_dir / f"{original_name}_transkript_{counter}.vtt"
        counter += 1

    shutil.copy(vtt_path, dest_path)
    return {"saved_to": str(dest_path), "filename": dest_path.name}


@app.post("/api/jobs/{job_id}/save-csv")
async def save_csv_to_downloads(job_id: str):
    """Save CSV file to user's Downloads folder"""
    import shutil

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    csv_path = Path(job.get("csv_path", ""))
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")

    # Save to Downloads folder
    downloads_dir = Path.home() / "Downloads"
    original_name = Path(job["filename"]).stem
    dest_path = downloads_dir / f"{original_name}_sprecher.csv"

    # Avoid overwriting
    counter = 1
    while dest_path.exists():
        dest_path = downloads_dir / f"{original_name}_sprecher_{counter}.csv"
        counter += 1

    shutil.copy(csv_path, dest_path)
    return {"saved_to": str(dest_path), "filename": dest_path.name}


@app.post("/api/jobs/{job_id}/save-txt")
async def save_txt_to_downloads(job_id: str):
    """Save plain-text file to user's Downloads folder"""
    import shutil

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    txt_path = Path(job.get("txt_path", ""))
    if not txt_path.exists():
        raise HTTPException(status_code=404, detail="TXT file not found")

    downloads_dir = Path.home() / "Downloads"
    original_name = Path(job["filename"]).stem
    dest_path = downloads_dir / f"{original_name}_transkript.txt"

    counter = 1
    while dest_path.exists():
        dest_path = downloads_dir / f"{original_name}_transkript_{counter}.txt"
        counter += 1

    shutil.copy(txt_path, dest_path)
    return {"saved_to": str(dest_path), "filename": dest_path.name}


def _pick_clean_window(diarization: list[dict], speaker_label: str,
                       clip_len: float = 10.0, margin: float = 1.0,
                       min_segment: float = 6.0,
                       min_gap: float = 0.3) -> tuple[float, float] | None:
    """
    Find a (start, end) inside a clean segment of the given speaker.

    Strategy:
      1. Sort all segments by start, locate the speaker's segments.
      2. Prefer segments that are >= min_segment seconds long AND have at
         least min_gap seconds of silence (or no neighbor) before/after.
      3. Among preferred candidates, pick the longest. Fallback: longest
         overall segment for that speaker.
      4. Extract a clip_len window from the middle with `margin` seconds
         buffer on both sides if the segment is long enough.
    """
    sorted_segs = sorted(diarization, key=lambda s: s["start"])
    own = [(i, s) for i, s in enumerate(sorted_segs) if s.get("speaker") == speaker_label]
    if not own:
        return None

    def neighbor_gap_before(idx: int) -> float:
        if idx == 0:
            return float("inf")
        prev = sorted_segs[idx - 1]
        if prev.get("speaker") == speaker_label:
            return float("inf")
        return sorted_segs[idx]["start"] - prev["end"]

    def neighbor_gap_after(idx: int) -> float:
        if idx >= len(sorted_segs) - 1:
            return float("inf")
        nxt = sorted_segs[idx + 1]
        if nxt.get("speaker") == speaker_label:
            return float("inf")
        return nxt["start"] - sorted_segs[idx]["end"]

    preferred = [
        (i, s) for i, s in own
        if (s["end"] - s["start"]) >= min_segment
        and neighbor_gap_before(i) >= min_gap
        and neighbor_gap_after(i) >= min_gap
    ]

    pool = preferred or own
    chosen = max(pool, key=lambda pair: pair[1]["end"] - pair[1]["start"])[1]

    seg_start, seg_end = chosen["start"], chosen["end"]
    duration = seg_end - seg_start

    # Apply margin only if the segment is long enough to absorb it
    if duration > clip_len + 2 * margin:
        inner_start = seg_start + margin
        inner_end = seg_end - margin
        mid = (inner_start + inner_end) / 2.0
        start = max(inner_start, mid - clip_len / 2.0)
        end = start + clip_len
    else:
        # Take whatever fits; still try to center
        clip = min(clip_len, duration)
        mid = (seg_start + seg_end) / 2.0
        start = max(seg_start, mid - clip / 2.0)
        end = start + clip

    return (start, end)


@app.get("/api/jobs/{job_id}/speaker/{speaker_label}/sample")
async def get_speaker_sample(job_id: str, speaker_label: str):
    """
    Stream a ~10-second WAV clip taken from a clean segment of the given
    speaker. Used by the correction modal to play a loopable preview.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    diar = job.get("diarization") or []
    if not any(seg.get("speaker") == speaker_label for seg in diar):
        raise HTTPException(status_code=404, detail="Speaker not in this job")

    window = _pick_clean_window(diar, speaker_label, clip_len=10.0)
    if not window:
        raise HTTPException(status_code=404, detail="No clean segment found for this speaker")
    start, end = window
    clip = end - start

    audio_src = job.get("upload_path")
    if not audio_src or not Path(audio_src).exists():
        raise HTTPException(status_code=404, detail="Source audio not found (already cleaned up)")

    outputs_dir = Path(job["output_path"]).parent
    sample_path = outputs_dir / f"{job_id}_sample_{speaker_label}.wav"
    if not sample_path.exists():
        import subprocess
        cmd = [
            _get_ffmpeg(), "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{clip:.3f}",
            "-i", str(audio_src),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(sample_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg failed: {result.stderr[:200]}")

    return FileResponse(sample_path, media_type="audio/wav")


def _sanitize_speaker_name(name: str) -> str:
    """Strip control chars / colons / line breaks; clamp length. Empty -> empty."""
    cleaned = "".join(c for c in (name or "") if c.isprintable() and c not in "\r\n\t:")
    return cleaned.strip()[:60]


class RenameSpeakersRequest(BaseModel):
    names: dict[str, str]


@app.post("/api/jobs/{job_id}/rename-speakers")
async def rename_speakers(job_id: str, request: RenameSpeakersRequest):
    """
    Apply a {speaker_label -> custom_name} map and re-render VTT/TXT/CSV
    so all references use the new names.

    Empty / missing entries fall back to the previously displayed name
    (default `Speaker 1`, `Speaker 2`, ... in input order).
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.get("status") != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    diar_data = job.get("diarization")
    if not diar_data:
        raise HTTPException(status_code=400, detail="Job has no diarization data")

    # Build the name map: caller-provided names override the defaults; blanks
    # revert to the auto-generated `Speaker N` label.
    from .merge import format_speaker_name as _default_name
    sanitized: dict[str, str] = {}
    for sp, raw in (request.names or {}).items():
        clean = _sanitize_speaker_name(raw)
        sanitized[sp] = clean if clean else _default_name(sp)
    # Fill in any speakers the client didn't send
    for sp in job.get("speakers", []):
        sanitized.setdefault(sp, _default_name(sp))
    job["speaker_names"] = sanitized

    # Re-render outputs by passing through the renamed labels. We rebuild from
    # the stored transcripts. all_segments isn't kept on the job — we have to
    # rebuild from diarization + saved files. Easier: read existing VTT, do a
    # textual prefix replacement of `OLD_NAME:` -> `NEW_NAME:` at each line
    # start. Same for TXT. CSV: replace whole-cell match in the Speaker column.
    vtt_path = Path(job["output_path"])
    txt_path = Path(job.get("txt_path", ""))
    csv_path = Path(job.get("csv_path", ""))

    # Map of OLD displayed name -> NEW displayed name
    rename_pairs: list[tuple[str, str]] = []
    for sp in job.get("speakers", []):
        old_label = _default_name(sp)
        new_label = sanitized[sp]
        if old_label != new_label:
            rename_pairs.append((old_label, new_label))

    if rename_pairs:
        for path in (vtt_path, txt_path):
            if not path or not Path(path).exists():
                continue
            text = Path(path).read_text(encoding="utf-8")
            for old, new in rename_pairs:
                # Match "OLD: " at start of a cue line. Use re for safety.
                import re
                pattern = re.compile(rf"(^|\n){re.escape(old)}: ", flags=re.MULTILINE)
                text = pattern.sub(rf"\1{new}: ", text)
            Path(path).write_text(text, encoding="utf-8")

        if csv_path and Path(csv_path).exists():
            import csv as csv_mod
            from io import StringIO
            rows = list(csv_mod.reader(Path(csv_path).read_text(encoding="utf-8").splitlines()))
            if rows:
                header = rows[0]
                speaker_col = header.index("Speaker") if "Speaker" in header else 2
                rename_map = dict(rename_pairs)
                for r in rows[1:]:
                    if len(r) > speaker_col and r[speaker_col] in rename_map:
                        r[speaker_col] = rename_map[r[speaker_col]]
                buf = StringIO()
                w = csv_mod.writer(buf, quoting=csv_mod.QUOTE_ALL)
                for r in rows:
                    w.writerow(r)
                Path(csv_path).write_text(buf.getvalue(), encoding="utf-8")

    return {
        "status": "ok",
        "speaker_names": sanitized,
        "renamed": dict(rename_pairs),
        "output_path": str(vtt_path),
        "csv_path": str(csv_path) if csv_path else None,
        "txt_path": str(txt_path) if txt_path else None,
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Delete files
    for path_key in ["upload_path", "output_path", "csv_path", "txt_path"]:
        path = Path(job.get(path_key, ""))
        if path.exists():
            path.unlink()

    del jobs[job_id]
    return {"status": "deleted"}


# Serve frontend
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
