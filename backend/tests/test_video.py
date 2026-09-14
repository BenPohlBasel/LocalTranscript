"""Video als Quelle (BACKLOG 8): keine Umwandlung, Codec-Abweisung mit
Hinweis, Ton nach mp3, Video unverändert im Eintrag, Video-Route mit
Range, qdpx wahlweise als VideoSource. Braucht ffmpeg (lavfi-Fixtures)."""
from __future__ import annotations

import io
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from turnscript import bibliothek, config, exporte, video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg fehlt")

VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n<v Anna>Hallo.\n\n00:00:01.500 --> 00:00:03.000\n<v Ben>Guten Tag.\n"


def _fixture(tmp_path: Path, name: str, vcodec: str, acodec: str | None = "aac") -> Path:
    p = tmp_path / name
    cmd = [config.get_ffmpeg_cli(), "-v", "quiet", "-y", "-f", "lavfi",
           "-i", "testsrc=size=160x90:rate=10"]
    if acodec:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440"]
    cmd += ["-t", "3", "-c:v", vcodec, "-pix_fmt", "yuv420p"]
    cmd += (["-c:a", acodec, "-shortest"] if acodec else ["-an"]) + [str(p)]
    subprocess.run(cmd, check=True)
    return p


def test_h264_mp4_wird_angenommen_vp9_abgewiesen(tmp_path):
    mp4 = _fixture(tmp_path, "a.mp4", "libx264")
    info = video.pruefe(mp4)
    assert info["video_codec"] == "h264" and info["audio_codec"] == "aac"
    assert info["breite"] == 160 and 2.5 < info["dauer_s"] < 3.5
    webm = _fixture(tmp_path, "b.webm", "libvpx-vp9", None)
    with pytest.raises(video.VideoFehler) as e:
        video.pruefe(webm)
    assert e.value.code == "codec" and "HandBrake" in str(e.value)
    stumm = _fixture(tmp_path, "c.mp4", "libx264", None)
    with pytest.raises(video.VideoFehler) as e:
        video.pruefe(stumm)
    assert e.value.code == "ohne-ton"


def test_reines_audio_ist_kein_video(tmp_path):
    mp3 = tmp_path / "t.mp3"
    subprocess.run([config.get_ffmpeg_cli(), "-v", "quiet", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440", "-t", "1", str(mp3)], check=True)
    assert video.pruefe(mp3) is None


def test_zu_gross_wird_abgewiesen(tmp_path, monkeypatch):
    mp4 = _fixture(tmp_path, "a.mp4", "libx264")
    monkeypatch.setattr(video, "VIDEO_MAX_BYTES", 10)
    with pytest.raises(video.VideoFehler) as e:
        video.pruefe(mp4)
    assert e.value.code == "zu-gross"


def test_import_mit_video_zieht_ton_und_behaelt_video(client, tmp_path):
    """VTT + Video: der Eintrag hat audio.mp3 (Arbeitskopie) UND das
    unveränderte video.mp4; enrich-Export nimmt nur den Ton."""
    mp4 = _fixture(tmp_path, "interview.mp4", "libx264")
    r = client.post("/api/import", files={
        "datei": ("interview.vtt", VTT.encode(), "text/vtt"),
        "audio": ("interview.mp4", mp4.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    eid = r.json()["eintrag"]
    d = bibliothek.lese(eid)
    assert d["audio"] == "audio.mp3" and d["video"] == "video.mp4"
    assert bibliothek.video_pfad(eid).read_bytes() == mp4.read_bytes()   # unverändert
    assert bibliothek.audio_pfad(eid).stat().st_size > 1000
    assert next(e for e in bibliothek.liste() if e["id"] == eid)["video"] is True
    # Video-Route mit Range (der WebView springt so an die Stelle)
    r = client.get(f"/api/transcripts/{eid}/video", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert r.headers["content-type"] == "video/mp4"
    # enrich: nur Ton
    inhalt, _n, _m = exporte.export_bytes(eid, "enrich")
    namen = zipfile.ZipFile(io.BytesIO(inhalt)).namelist()
    assert any(n.endswith("/source/audio.mp3") for n in namen)
    assert not any(n.endswith(".mp4") for n in namen)
    # qdpx: wahlweise mit Video (VideoSource + Datei im Media-Ordner)
    ziel = tmp_path / "aus.qdpx.zip"
    assert exporte.export_nach(eid, "qdpx-video", ziel) == "interview.qdpx.zip"
    aussen = zipfile.ZipFile(ziel)                               # gestreamt, nie als bytes
    assert any(n.endswith(".mp4") for n in aussen.namelist())
    qde = zipfile.ZipFile(io.BytesIO(aussen.read("interview.qdpx"))).read("interview.qde").decode()
    assert "<VideoSource" in qde and "<AudioSource" not in qde and "SyncPoint" in qde
    inhalt, _n, _m = exporte.export_bytes(eid, "qdpx")
    aussen = zipfile.ZipFile(io.BytesIO(inhalt))
    assert any(n.endswith(".mp3") for n in aussen.namelist())


def test_ton_container_ohne_bild_ist_kein_video(client, tmp_path):
    """Eine .mp4 mit nur Tonspur (m4a-artig) wird zur mp3 — kein
    video.mp4, kein schwarzes Bild, kein «mit Video»-Export."""
    nur_ton = tmp_path / "ton.mp4"
    subprocess.run([config.get_ffmpeg_cli(), "-v", "quiet", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440", "-t", "2", "-c:a", "aac", str(nur_ton)], check=True)
    assert video.pruefe(nur_ton) is None
    r = client.post("/api/import", files={
        "datei": ("ton.vtt", VTT.encode(), "text/vtt"),
        "audio": ("ton.mp4", nur_ton.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    d = bibliothek.lese(r.json()["eintrag"])
    assert d["audio"] == "audio.mp3" and not d.get("video")
    r = client.get(f"/api/transcripts/{d['id']}/export/qdpx-video")
    assert r.status_code == 409


def test_fremder_container_wird_vor_dem_job_abgewiesen(client, tmp_path):
    mp4 = _fixture(tmp_path, "vortrag.mp4", "libx264")
    mkv = tmp_path / "vortrag.mkv"
    subprocess.run([config.get_ffmpeg_cli(), "-v", "quiet", "-y", "-i", str(mp4),
                    "-c", "copy", str(mkv)], check=True)
    r = client.post("/api/transcribe-path", json={"path": str(mkv)})
    assert r.status_code == 400, r.text                    # Endung nicht in der Liste
    webm = _fixture(tmp_path, "vortrag.webm", "libvpx-vp9", "libopus")
    r = client.post("/api/transcribe", files={"file": ("vortrag.webm", webm.read_bytes(), "video/webm")})
    assert r.status_code == 422 and "HandBrake" in r.text    # webm mit Bild: Codec-Hinweis
