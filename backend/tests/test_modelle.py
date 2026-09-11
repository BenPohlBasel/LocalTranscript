"""Eigene Whisper-Modelle aus `<Bibliothek>/Modelle/` (User 2026-09-11):
still ablegen, erscheint in der Auswahl; geprüft am ggml-Magic, halb
kopierte Dateien werden nicht angeboten; eigen schlägt mitgeliefert."""
from __future__ import annotations

import os
import time
from pathlib import Path

from localtranscript import config


def _modell(p: Path, groesse: int = 4096, alt: bool = True,
            magic: bytes = config.GGML_MAGIC) -> Path:
    p.write_bytes(magic + b"\0" * (groesse - 4))
    if alt:   # «fertig kopiert»: mtime ausserhalb der Ruhezeit
        t = time.time() - 60
        os.utime(p, (t, t))
    return p


def test_eigene_modelle_erscheinen_und_werden_geprueft(client, tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"; bundle.mkdir()
    _modell(bundle / "ggml-medium.bin", 8192)
    monkeypatch.setenv("LT_MODELS_DIR", str(bundle))
    r = client.get("/api/models")
    assert r.status_code == 200
    eigene = Path(r.json()["eigene_dir"])
    assert eigene.is_dir() and eigene.name == "Modelle"      # angelegt
    _modell(eigene / "ggml-large-v3-turbo.bin", 16384)
    _modell(eigene / "ggml-medium.bin", 2048)                # ersetzt das Bundle
    _modell(eigene / "ggml-frisch.bin", alt=False)           # wird noch kopiert
    _modell(eigene / "ggml-falsch.bin", magic=b"%PDF")       # kein Modell
    (eigene / "notizen.txt").write_text("x")
    m = client.get("/api/models").json()
    namen = {x["name"]: x for x in m["models"]}
    assert set(namen) == {"medium", "large-v3-turbo"}
    assert namen["medium"]["quelle"] == "eigen" and namen["medium"]["size_mb"] == 0.0
    assert namen["large-v3-turbo"]["quelle"] == "eigen"
    assert {(u["datei"], u["grund"]) for u in m["ungueltig"]} == {
        ("ggml-frisch.bin", "kopiert"), ("ggml-falsch.bin", "kein-ggml"),
        ("notizen.txt", "name")}
    # der Lauf nimmt dieselbe Datei wie die Auswahl
    assert config.model_pfad("medium") == eigene / "ggml-medium.bin"
    assert config.model_pfad("large-v3-turbo").parent == eigene


def test_fehlendes_modell_ist_ein_klarer_fehler(client, tmp_path, monkeypatch):
    monkeypatch.setenv("LT_MODELS_DIR", str(tmp_path / "leer"))
    try:
        config.model_pfad("gibtsnicht")
    except FileNotFoundError as e:
        assert "ggml-gibtsnicht.bin" in str(e)
    else:
        raise AssertionError("kein Fehler")
