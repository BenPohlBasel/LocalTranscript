"""Job-Pipeline mit Fakes (kein ffmpeg/whisper/torch nötig):
diarisierter Weg End-zu-Ende bis zum Bibliotheks-Eintrag, Abbruch."""
from __future__ import annotations

import time


def _warte(client, job_id: str, bis: str, timeout: float = 10.0) -> dict:
    frist = time.time() + timeout
    while time.time() < frist:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in (bis, "failed"):
            return j
        time.sleep(0.02)
    raise AssertionError(f"Timeout — zuletzt: {j}")


def test_diarize_pipeline_landet_in_bibliothek(client, tmp_path,
                                               monkeypatch):
    from localtranscript import jobs
    from localtranscript.diarize import SpeakerSegment
    from localtranscript.transcribe import TranscriptSegment

    audio = tmp_path / "interview.mp3"
    audio.write_bytes(b"ID3fake")

    monkeypatch.setattr(jobs, "_konvertiere",
                        lambda job, quelle, d: quelle)
    def fake_clip(job, wav, a, b, i):
        # ECHTE eigene Datei — der Code löscht Clips nach Gebrauch,
        # die Quelle darf das nie treffen
        c = wav.parent / f"clip{i}.wav"
        c.write_bytes(b"RIFFfake")
        return c
    monkeypatch.setattr(jobs, "_clip", fake_clip)
    import localtranscript.diarize as dia
    # `fortschritt` spiegelt die echte Signatur — der Job meldet damit
    # den Diarisierungs-Fortschritt (10 → 25 %); der Doppelgänger ruft
    # ihn einmal, damit der Meldeweg mitgeprüft ist.
    def fake_diarize(pfad, mi, ma, th, fortschritt=None):
        if fortschritt is not None:
            fortschritt(0, 2)
        return [SpeakerSegment(0.0, 5.0, "SPEAKER_01"),
                SpeakerSegment(5.0, 20.0, "SPEAKER_00")]
    monkeypatch.setattr(dia, "diarize_audio", fake_diarize)

    def fake_seg(clip, model, lang, time_offset=0.0, register=None):
        return [TranscriptSegment(time_offset, time_offset + 2.0,
                                  f"Text ab {time_offset:.0f}.")]
    monkeypatch.setattr(jobs, "transcribe_segment", fake_seg)

    r = client.post("/api/transcribe-path",
                    json={"path": str(audio), "language": "de"})
    assert r.status_code == 200, r.text
    j = _warte(client, r.json()["job_id"], "completed")
    assert j["status"] == "completed", j
    d = client.get(f"/api/transcripts/{j['eintrag']}").json()
    # SPEAKER_00 spricht länger ⇒ wird "Sprecher 1"
    namen = {s["id"]: s["name"] for s in d["sprecher"]}
    seg = d["segmente"]
    assert len(seg) == 2
    assert namen[seg[0]["sprecher"]] == "Sprecher 2"
    assert namen[seg[1]["sprecher"]] == "Sprecher 1"
    assert d["audio"] == "audio.mp3"
    assert d["quelle"]["erzeugt"] == "transcription"
    # Original bleibt liegen (Pfad-Weg kopiert, löscht nie)
    assert audio.is_file()


def test_abbruch(client, tmp_path, monkeypatch):
    from localtranscript import jobs

    audio = tmp_path / "lang.mp3"
    audio.write_bytes(b"ID3fake")

    def langsam(job, quelle, d):
        for _ in range(200):
            jobs._pruefe_abbruch(job)
            time.sleep(0.02)
        return quelle
    monkeypatch.setattr(jobs, "_konvertiere", langsam)
    r = client.post("/api/transcribe-path", json={"path": str(audio)})
    job_id = r.json()["job_id"]
    time.sleep(0.1)
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    j = _warte(client, job_id, "cancelled")
    assert j["status"] == "cancelled"
