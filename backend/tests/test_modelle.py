"""Eigene Whisper-Modelle aus `<Bibliothek>/Modelle/` (User 2026-09-11):
still ablegen, erscheint in der Auswahl; geprüft am ggml-Magic, halb
kopierte Dateien werden nicht angeboten; eigen schlägt mitgeliefert."""
from __future__ import annotations

import os
import time
from pathlib import Path

from localtranscript import config


def _modell(p: Path, groesse: int = 40_000_000, alt: bool = True,
            magic: bytes = config.GGML_MAGIC) -> Path:
    p.write_bytes(magic + b"\0" * (groesse - 4))
    if alt:   # «fertig kopiert»: mtime ausserhalb der Ruhezeit
        t = time.time() - 60
        os.utime(p, (t, t))
    return p


def test_eigene_modelle_erscheinen_und_werden_geprueft(client, tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"; bundle.mkdir()
    _modell(bundle / "ggml-medium.bin", 41_000_000)
    monkeypatch.setenv("LT_MODELS_DIR", str(bundle))
    r = client.get("/api/models")
    assert r.status_code == 200
    eigene = Path(r.json()["eigene_dir"])
    assert eigene.is_dir() and eigene.name == "Modelle"      # angelegt
    _modell(eigene / "ggml-large-v3-turbo.bin", 42_000_000)
    _modell(eigene / "ggml-medium.bin", 40_000_000)          # ersetzt das Bundle
    _modell(eigene / "ggml-abgebrochen.bin", 600_000)        # Kopf + Tokens, kein Modell
    _modell(eigene / "ggml-frisch.bin", alt=False)           # wird noch kopiert
    _modell(eigene / "ggml-falsch.bin", magic=b"%PDF")       # kein Modell
    (eigene / "notizen.txt").write_text("x")
    m = client.get("/api/models").json()
    namen = {x["name"]: x for x in m["models"]}
    assert set(namen) == {"medium", "large-v3-turbo"}
    assert namen["medium"]["quelle"] == "eigen" and namen["medium"]["size_mb"] == 40.0
    assert namen["large-v3-turbo"]["quelle"] == "eigen"
    assert {(u["datei"], u["grund"]) for u in m["ungueltig"]} == {
        ("ggml-frisch.bin", "kopiert"), ("ggml-falsch.bin", "kein-ggml"),
        ("ggml-abgebrochen.bin", "unvollstaendig"), ("notizen.txt", "name")}
    # der Lauf nimmt dieselbe Datei wie die Auswahl
    assert config.model_pfad("medium") == eigene / "ggml-medium.bin"
    assert config.model_pfad("large-v3-turbo").parent == eigene
    # … auch wenn die eigene Datei durchfällt: dann gilt das Bundle
    _modell(eigene / "ggml-medium.bin", magic=b"%PDF")
    assert config.model_pfad("medium") == bundle / "ggml-medium.bin"
    assert {x["name"]: x["quelle"] for x in client.get("/api/models").json()["models"]}["medium"] == "bundled"


def test_ausgelagerte_icloud_datei_wird_nicht_geoeffnet(client, tmp_path, monkeypatch):
    """SF_DATALESS (Inhalt nur in iCloud): nicht anbieten, nicht öffnen —
    open() würde den Scan blockieren, bis iCloud die Datei geholt hat."""
    import os
    monkeypatch.setenv("LT_MODELS_DIR", str(tmp_path / "leer"))
    eigene = Path(client.get("/api/models").json()["eigene_dir"])
    p = _modell(eigene / "ggml-wolke.bin")
    echt = os.stat
    class _St:
        def __init__(self, st): self._st = st; self.st_flags = config.SF_DATALESS
        def __getattr__(self, n): return getattr(self._st, n)
    monkeypatch.setattr(Path, "stat", lambda self, **k: _St(echt(self)) if self == p else echt(self, **k))
    geoeffnet = []
    echt_open = Path.open
    def _open(self, *a, **k):
        if self == p:
            geoeffnet.append(self); raise AssertionError("geöffnet")
        return echt_open(self, *a, **k)
    monkeypatch.setattr(Path, "open", _open)
    m = client.get("/api/models").json()
    assert next(u for u in m["ungueltig"] if u["datei"] == "ggml-wolke.bin")["grund"] == "icloud"
    assert not [x for x in m["models"] if x["name"] == "wolke"] and not geoeffnet


def test_fehlendes_modell_ist_ein_klarer_fehler(client, tmp_path, monkeypatch):
    monkeypatch.setenv("LT_MODELS_DIR", str(tmp_path / "leer"))
    try:
        config.model_pfad("gibtsnicht")
    except FileNotFoundError as e:
        assert "ggml-gibtsnicht.bin" in str(e)
    else:
        raise AssertionError("kein Fehler")
