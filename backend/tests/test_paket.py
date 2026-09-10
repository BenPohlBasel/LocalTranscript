"""Weitergabe-Rundlauf: .enrich exportieren und wieder einlesen.

Regel (User 2026-09-09): gezogen werden NUR `transkript.json` und das
Audio; jede Analyse-Schicht fällt weg — nach dem ersten Edit stimmte
ohnehin keine mehr. Fehlt die Beilage, bleibt die Zeitkarte als
gröberer Rückfall.
"""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from localtranscript import bibliothek, exporte, paket


def test_rundlauf_ist_verlustfrei(client, eintrag):
    """Export → lies() gibt Segmente und Sprecher UNVERÄNDERT zurück."""
    orig = bibliothek.lese(eintrag)
    inhalt, _name, _mt = exporte.export_bytes(eintrag, "enrich")
    p = paket.lies(inhalt)
    assert p["genau"] is True
    assert p["segmente"] == orig["segmente"]
    assert p["sprecher"] == orig["sprecher"]
    # die Segment-ids überleben — sie sind die Identität im Modell
    assert [s["id"] for s in p["segmente"]] == \
           [s["id"] for s in orig["segmente"]]


def test_beilage_liegt_im_dossier_ohne_layer_zu_sein(client, eintrag):
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    z = zipfile.ZipFile(io.BytesIO(inhalt))
    tj = next(n for n in z.namelist()
              if n.endswith("/transkript.json"))
    manifest = next(n for n in z.namelist() if n.endswith("/manifest.json"))
    current = json.loads(z.read(manifest))["current"]
    # Beilage wie audio.mp3 und source.pdf: im Ordner, NICHT im Manifest
    assert "transkript.json" not in current
    assert json.loads(z.read(tj))["segmente"]


def test_import_endpunkt_legt_eintrag_an(client, eintrag):
    inhalt, name, _m = exporte.export_bytes(eintrag, "enrich")
    r = client.post("/api/import",
                    files={"datei": (name, inhalt, "application/zip")})
    assert r.status_code == 200, r.text
    assert r.json()["genau"] is True
    neu = bibliothek.lese(r.json()["eintrag"])
    orig = bibliothek.lese(eintrag)
    assert neu["segmente"] == orig["segmente"]
    assert neu["sprecher"] == orig["sprecher"]
    assert neu["id"] != orig["id"]          # neuer Eintrag, nicht Ersatz
    assert neu["quelle"]["erzeugt"] == "import-enrich"


def _ohne(inhalt: bytes, blatt: str) -> bytes:
    """Dasselbe Zip ohne eine bestimmte Datei."""
    alt = zipfile.ZipFile(io.BytesIO(inhalt))
    aus = io.BytesIO()
    with zipfile.ZipFile(aus, "w", zipfile.ZIP_DEFLATED) as neu:
        for i in alt.infolist():
            if i.filename.rsplit("/", 1)[-1] != blatt:
                neu.writestr(i, alt.read(i.filename))
    return aus.getvalue()


def test_rueckfall_auf_die_zeitkarte(client, eintrag):
    """Ohne Beilage: gröber, aber MIT Wortlaut — leere Segmente wären
    schlimmer als eine Absage."""
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    p = paket.lies(_ohne(inhalt, "transkript.json"))
    assert p["genau"] is False
    assert p["segmente"]
    assert all(s["text"].strip() for s in p["segmente"])
    assert p["sprecher"]


def test_ohne_zeitkarte_und_beilage_klare_absage(client, eintrag):
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    roh = _ohne(_ohne(inhalt, "transkript.json"), "2z-zeitkarte.json")
    with pytest.raises(paket.PaketFehler, match="transkript.json"):
        paket.lies(roh)


def test_kein_zip_wird_abgewiesen():
    with pytest.raises(paket.PaketFehler, match="Zip"):
        paket.lies(b"WEBVTT\n\nkein Zip")


def test_audio_kommt_mit(client, eintrag):
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    p = paket.lies(inhalt)
    # die Fixture hat kein Audio — dann fehlt es ehrlich, ohne Fehler
    assert p["audio_name"] is None or p["audio_bytes"]


# ---------- Weitergabeform 2026-09-10: EINE Datei .enrich, unkomprimiert ----------

def test_export_heisst_enrich_und_ist_unkomprimiert(eintrag):
    """Endung .enrich (nicht .enrich.zip), ZIP_STORED für jedes Mitglied,
    genau ein Wurzelverzeichnis — so erwartet es enrich.unpack."""
    inhalt, name, mime = exporte.export_bytes(eintrag, "enrich")
    assert name.endswith(".enrich") and not name.endswith(".zip")
    assert mime == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(inhalt))
    assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
    wurzeln = {n.split("/", 1)[0] for n in z.namelist()}
    assert len(wurzeln) == 1 and next(iter(wurzeln)).endswith(".enrich")


def test_import_flache_enrich_datei_per_upload(client, eintrag):
    inhalt, name, _m = exporte.export_bytes(eintrag, "enrich")
    r = client.post("/api/import",
                    files={"datei": (name, inhalt, "application/zip")})
    assert r.status_code == 200, r.text
    assert r.json()["genau"] is True


def test_import_enrich_als_verzeichnis(client, eintrag, tmp_path):
    """Das Dossier, wie enrich es ablegt — ein Verzeichnis (auf dem Mac
    ein Package). Entpackt, dann über import-path als Pfad."""
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    zipfile.ZipFile(io.BytesIO(inhalt)).extractall(tmp_path)
    ordner = next(tmp_path.glob("*.enrich"))
    assert ordner.is_dir()
    r = client.post("/api/import-path", json={"path": str(ordner)})
    assert r.status_code == 200, r.text
    neu = bibliothek.lese(r.json()["eintrag"])
    assert neu["segmente"] == bibliothek.lese(eintrag)["segmente"]
    assert neu["quelle"]["erzeugt"] == "import-enrich"


def test_import_altes_enrich_zip_geht_weiter(client, eintrag, tmp_path):
    """Was frühere Versionen exportierten (.enrich.zip), bleibt still
    lesbar — es wird nur nicht mehr angeboten."""
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    alt = tmp_path / "alt.enrich.zip"; alt.write_bytes(inhalt)
    r = client.post("/api/import-path", json={"path": str(alt)})
    assert r.status_code == 200, r.text


def test_enrich_kann_die_datei_entpacken(eintrag, tmp_path):
    """Gegenprobe mit enrich selbst: Dossier.unpack nimmt die flache
    .enrich an — sie ist ein Zip am Inhalt, nicht am Namen."""
    from enrich_core.dossier import Dossier
    inhalt, name, _m = exporte.export_bytes(eintrag, "enrich")
    datei = tmp_path / name; datei.write_bytes(inhalt)
    d = Dossier.unpack(datei, tmp_path / "aus")
    assert (d.path / "transkript.json").is_file()
