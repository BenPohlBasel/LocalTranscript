"""Test-Fixtures: isolierte Config + Bibliothek je Test (LT_CONFIG_DIR
zeigt auf tmp, library_root wird gesetzt) — kein Test berührt echte
User-Daten. ffmpeg/whisper werden NIE echt gebraucht (Fakes)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LT_CONFIG_DIR", str(tmp_path / "cfg"))
    from localtranscript import config
    from localtranscript.main import app
    lib = tmp_path / "bibliothek"
    lib.mkdir()
    config.write_config({"library_root": str(lib)})
    return TestClient(app)


VTT = """WEBVTT

1
00:00:00.000 --> 00:00:04.000
Anna: Hallo und willkommen zur Sitzung.

2
00:00:04.000 --> 00:00:07.500
Ben: Danke, schön hier zu sein.

3
00:00:07.500 --> 00:00:11.000
Weiter geht es mit dem zweiten Punkt.
"""


@pytest.fixture()
def eintrag(client):
    """Ein importiertes Transkript (3 Cues, Cue 3 setzt Bens Turn fort)."""
    r = client.post("/api/import",
                    files={"datei": ("probe.vtt", VTT.encode(),
                                     "text/vtt")})
    assert r.status_code == 200, r.text
    return r.json()["eintrag"]
