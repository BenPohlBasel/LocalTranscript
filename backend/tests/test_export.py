"""Exporte: vtt/csv/txt-Form + das enrich-Dossier-Zip End-zu-Ende
(mit enrich_core zurückgelesen: T1, Zeitkarte, narrative Kette)."""
from __future__ import annotations

import zipfile


def test_vtt_roundtrip(client, eintrag):
    r = client.get(f"/api/transcripts/{eintrag}/export/vtt")
    assert r.status_code == 200
    text = r.content.decode("utf-8")
    assert text.startswith("WEBVTT")
    # WebVTT-Standard: Sprecher als Voice-Tag, nie im Text
    assert "<v Anna>" in text and "<v Ben>" in text
    assert "Anna: " not in text
    # Reimport des eigenen Exports ergibt dieselben Turns
    r2 = client.post("/api/import",
                     files={"datei": ("re.vtt", r.content, "text/vtt")})
    d = client.get(f"/api/transcripts/{r2.json()['eintrag']}").json()
    assert len(d["segmente"]) == 2
    assert [s["name"] for s in d["sprecher"]] == ["Anna", "Ben"]


def test_csv_immer_hhmmss(client, eintrag):
    text = client.get(
        f"/api/transcripts/{eintrag}/export/csv").content.decode(
        "utf-8-sig")
    zeilen = text.strip().splitlines()
    assert zeilen[0] == '"Time-in","Time-out","Speaker","Text"'
    assert '"00:00:00"' in zeilen[1]  # hh:mm:ss auch unter 1 h


def test_txt(client, eintrag):
    text = client.get(
        f"/api/transcripts/{eintrag}/export/txt").content.decode()
    assert text.startswith("Anna: Hallo")
    assert "\n\nBen: " in text


def test_enrich_export_ist_echtes_dossier(client, eintrag, tmp_path):
    r = client.get(f"/api/transcripts/{eintrag}/export/enrich")
    assert r.status_code == 200, r.text
    zp = tmp_path / "e.enrich.zip"
    zp.write_bytes(r.content)
    assert zipfile.is_zipfile(zp)

    from enrich_core.dossier import Dossier
    d = Dossier.unpack(zp, tmp_path / "aus")
    m = d.manifest
    assert m.analyse_kette == "narrativ"
    assert "2z-zeitkarte.json" in m.current
    t1 = d.read_layer("3-text-clean.json")
    haupt = t1.streams["main"]
    # Sprecher fett + Timecode hh:mm:ss stehen im T1 (enrich-Konvention)
    assert "Anna [00:00:00]: Hallo und willkommen" in haupt
    assert "Ben [00:00:04]: Danke" in haupt
    zk = d.read_layer("2z-zeitkarte.json")
    assert len(zk.einheiten) == 2
    assert zk.einheiten[1].speaker == "Ben"
    # PDF liegt und ist eines
    pdf = d.path / "source.pdf"
    assert pdf.is_file() and pdf.read_bytes()[:5] == b"%PDF-"


def test_export_in_datei(client, eintrag, tmp_path):
    ziel = tmp_path / "sitzung.vtt"
    r = client.post(f"/api/transcripts/{eintrag}/export",
                    json={"format": "vtt", "path": str(ziel)})
    assert r.status_code == 200
    assert ziel.read_text("utf-8").startswith("WEBVTT")


def test_export_endung_muss_passen(client, eintrag, tmp_path):
    ziel = tmp_path / "zshrc"  # falsche Endung fürs Format
    r = client.post(f"/api/transcripts/{eintrag}/export",
                    json={"format": "txt", "path": str(ziel)})
    assert r.status_code == 409 and "enden" in r.json()["detail"]


def test_enrich_export_konvertiert_wav_zu_mp3(client, tmp_path):
    """User-Regel: im .enrich liegt immer mp3, nie wav."""
    import shutil
    import wave

    import pytest
    if shutil.which("ffmpeg") is None:
        pytest.skip("kein ffmpeg")
    w = tmp_path / "ton.wav"
    with wave.open(str(w), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 16000)
    r = client.post("/api/import", files={
        "datei": ("t.vtt", VTT_MINI.encode(), "text/vtt"),
        "audio": ("ton.wav", w.read_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    eid = r.json()["eintrag"]
    r = client.get(f"/api/transcripts/{eid}/export/enrich")
    assert r.status_code == 200, r.text
    import zipfile
    namen = zipfile.ZipFile(__import__("io").BytesIO(r.content)).namelist()
    assert any(n.endswith("/audio.mp3") for n in namen), namen
    assert not any(n.endswith(".wav") for n in namen)


VTT_MINI = """WEBVTT

1
00:00:00.000 --> 00:00:01.000
Anna: Ton läuft.
"""
