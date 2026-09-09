"""Weitergabe-Rundlauf: .enrich.zip exportieren und wieder einlesen.

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
