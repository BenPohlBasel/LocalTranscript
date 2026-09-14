"""Umbenennung LocalTranscript → TurnScript (3.0.0): nichts beim Nutzer
geht verloren, und alte Dossiers bleiben die eigenen."""
from __future__ import annotations

import json

from turnscript import config
from turnscript.format2 import EIGENE_TOOLS, TOOL, _ist_unser_run


def test_einstellungen_werden_kopiert_nicht_verschoben(tmp_path):
    alt, neu = tmp_path / "LocalTranscript", tmp_path / "TurnScript"
    alt.mkdir()
    (alt / "config.json").write_text(json.dumps({"library_root": "/x/Bib"}))
    (alt / "app-backend.json").write_text("{}")
    config._alt_uebernehmen(alt, neu)
    assert json.loads((neu / "config.json").read_text())["library_root"] == "/x/Bib"
    assert not (neu / "app-backend.json").exists()      # fremder Prozess
    assert (alt / "config.json").exists()               # alte App bleibt benutzbar


def test_bestehende_neue_einstellungen_bleiben_unberuehrt(tmp_path):
    alt, neu = tmp_path / "LocalTranscript", tmp_path / "TurnScript"
    alt.mkdir(); neu.mkdir()
    (alt / "config.json").write_text("{}")
    (neu / "config.json").write_text(json.dumps({"library_root": "/neu"}))
    config._alt_uebernehmen(alt, neu)
    assert json.loads((neu / "config.json").read_text())["library_root"] == "/neu"


def test_alte_und_neue_runs_gelten_als_eigene():
    assert TOOL == "turnscript" and "localtranscript" in EIGENE_TOOLS
    assert _ist_unser_run({"who": {"app": "localtranscript"}})
    assert _ist_unser_run({"who": {"app": "localtranscript/2.5.0"}})
    assert _ist_unser_run({"tool": "localtranscript"})
    assert _ist_unser_run({"who": {"app": "turnscript"}})
    assert not _ist_unser_run({"who": {"app": "enrich"}})


def test_identitaet_nennt_den_neuen_namen():
    assert config.identitaet()["app"].startswith("turnscript/")
