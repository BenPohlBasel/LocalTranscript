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


def test_alter_name_ist_wirklich_der_alte():
    """Eine pauschale Ersetzung LocalTranscript → TurnScript hatte diese
    Konstante mitgenommen; die Übernahme suchte dann im NEUEN Ordner."""
    assert config.ALTER_NAME == "LocalTranscript"
    assert config.APP_NAME == "TurnScript"


def test_echter_ablauf_mit_heimverzeichnis(tmp_path, monkeypatch):
    """Ohne LT_CONFIG_DIR, mit den echten Ordnernamen unter einem
    Schein-HOME — so, wie die App beim ersten Start läuft."""
    monkeypatch.delenv("LT_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    alt = tmp_path / "Library" / "Application Support" / "LocalTranscript"
    for d in ("Cache", "GPUCache", "outputs"):
        (alt / d).mkdir(parents=True)
    (alt / "app-backend.json").write_text("{}")
    bib = tmp_path / "Documents" / "LocalTranscript"   # muss existieren,
    bib.mkdir(parents=True)                           # sonst setzt read_config ihn zurück
    (alt / "config.json").write_text(json.dumps({
        "library_root": str(bib),
        "install_id": "ins-ALT", "zotero_consent": True,
        "user_email": "a@b.ch"}))
    cfg = config.read_config()
    assert cfg["library_root"] == str(bib)
    assert cfg["install_id"] == "ins-ALT" and cfg["zotero_consent"] is True
    neu = tmp_path / "Library" / "Application Support" / "TurnScript"
    assert sorted(p.name for p in neu.iterdir()) == ["config.json"]
    assert (alt / "config.json").exists()


def test_leerer_neuer_ordner_blockiert_nicht(tmp_path):
    alt, neu = tmp_path / "LocalTranscript", tmp_path / "TurnScript"
    alt.mkdir(); neu.mkdir()
    (alt / "config.json").write_text(json.dumps({"library_root": "/x/Bib"}))
    config._alt_uebernehmen(alt, neu)
    assert json.loads((neu / "config.json").read_text())["library_root"] == "/x/Bib"


def test_bestehender_localtranscript_ordner_wird_vorgeschlagen(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Documents" / "LocalTranscript").mkdir(parents=True)
    assert config.default_library_root().name == "LocalTranscript"
    (tmp_path / "Documents" / "TurnScript").mkdir()
    assert config.default_library_root().name == "TurnScript"
