"""Exporte: vtt/csv/txt-Form + das enrich-Dossier-Zip End-zu-Ende
(mit enrich_core zurückgelesen: T1, Zeitkarte, narrative Kette)."""
from __future__ import annotations

import io
import json
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


def test_enrich_export_ist_format2_container(client, eintrag, tmp_path):
    """Der Export ist ein Format-2-Container, wie enrich ihn heute
    schreibt (Nummern, Wurzel): eine Wurzel, Inventar mit Hash für JEDE
    Datei, Lineage, das Transkript als Quelle, das PDF als Lesefassung.
    Der Inhalt der Text-Schichten ist der aus textsatz: main = reine
    Rede, Label-Zeilen im other-Strom."""
    r = client.get(f"/api/transcripts/{eintrag}/export/enrich")
    assert r.status_code == 200, r.text
    z = zipfile.ZipFile(io.BytesIO(r.content))
    wurzeln = {n.split("/", 1)[0] for n in z.namelist()}
    assert len(wurzeln) == 1 and next(iter(wurzeln)).endswith(".enrich")
    w = next(iter(wurzeln))
    m = json.loads(z.read(f"{w}/manifest.json"))
    assert m["analyse_kette"] == "narrativ" and m["profile"] == "handover"
    assert m["source"] == {"kind": "transcript", "canonical": "transcript.json",
                           "media": None, "rendered": "source.pdf"}
    assert m["producer"] == {"app": "localtranscript", "version": m["producer"]["version"]}
    t1 = json.loads(z.read(f"{w}/3-text-clean.json"))
    assert t1["kind"] == "text-clean" and t1["origin"] == "machine"
    haupt = t1["streams"]["main"]
    assert haupt.startswith("Hallo und willkommen")
    assert "Anna" not in haupt and "[00:" not in haupt
    other = t1["streams"].get("other", "")
    assert "Anna" in other and "[00:00:00]" in other
    zk = json.loads(z.read(f"{w}/2z-zeitkarte.json"))
    assert len(zk["einheiten"]) == 2 and zk["einheiten"][1]["speaker"] == "Ben"
    assert z.read(f"{w}/source.pdf")[:5] == b"%PDF-"
    assert m["files"]["source.pdf"]["role"] == "rendered"


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
