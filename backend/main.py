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
    from .transcribe import transcribe_segment_with_timestamps, transcribe_classic
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

        # Step 2: Transcribe each speaker segment with VTT timestamps
        job["status"] = JobStatus.TRANSCRIBING
        all_segments = []  # List of (speaker, transcript_segments)
        all_text_parts = []

        for i, diar_seg in enumerate(merged_diar):
            progress = 25 + int((i / len(merged_diar)) * 55)
            speaker_name = format_speaker_name(diar_seg.speaker)
            job["progress"] = progress
            job["message"] = f"Transkribiere Segment {i+1}/{len(merged_diar)} ({speaker_name})..."

            # Extract audio segment with ffmpeg
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
                str(segment_path)
            ]
            await asyncio.to_thread(
                subprocess.run, extract_cmd, capture_output=True
            )

            # Transcribe segment with timestamps (offset by segment start time)
            transcript_segments = await asyncio.to_thread(
                transcribe_segment_with_timestamps,
                str(segment_path),
                job["model"],
                job["language"],
                diar_seg.start  # Time offset
            )

            if transcript_segments:
                all_segments.append({
                    'speaker': diar_seg.speaker,
                    'segments': transcript_segments
                })
                # Update live preview
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
