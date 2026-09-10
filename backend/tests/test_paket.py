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


def test_transkript_ist_registrierte_quelle(client, eintrag):
    """Format 2: das Transkript ist die Quelle — eine Schicht mit Kopf,
    im Inventar mit Hash, nicht mehr eine Beilage ohne Eintrag."""
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    z = zipfile.ZipFile(io.BytesIO(inhalt))
    w = z.namelist()[0].split("/", 1)[0]
    m = json.loads(z.read(f"{w}/manifest.json"))
    f = m["files"]["source/transcript.json"]
    assert f["role"] == "layer" and f["layer"] in m["layers"]
    t = json.loads(z.read(f"{w}/source/transcript.json"))
    assert t["kind"] == "transcript" and t["schema"] == "transcript/1.0.0"
    assert t["segments"] and all("origin" in s for s in t["segments"])


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


def _format1(mit_beilage: bool, mit_zeitkarte: bool) -> bytes:
    """Ein Dossier, wie enrich und frühere LocalTranscript-Versionen es
    schrieben: Format 1, transkript.json als Beilage, Zeitkarte + T1."""
    aus = io.BytesIO()
    with zipfile.ZipFile(aus, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("alt.enrich/manifest.json", json.dumps(
            {"format": 1, "schema": "manifest/0.1.0", "current": {}, "runs": []}))
        if mit_beilage:
            z.writestr("alt.enrich/transkript.json", json.dumps(
                {"schema": 1, "name": "Alt", "sprecher": [{"id": "sp1", "name": "Anna"}],
                 "segmente": [{"id": "a", "start": 0.0, "end": 2.0,
                               "sprecher": "sp1", "text": "Hallo."}]}))
        if mit_zeitkarte:
            z.writestr("alt.enrich/3-text-clean.json", json.dumps(
                {"schema": "text-clean/0.1.0", "revision": "r1",
                 "streams": {"main": "Hallo."}}))
            z.writestr("alt.enrich/2z-zeitkarte.json", json.dumps(
                {"schema": "zeitkarte/0.1.0", "revision": "r1", "audio": "audio.mp3",
                 "einheiten": [{"start": 0, "end": 6, "speaker": "Anna",
                                "t0_s": 0.0, "t1_s": 2.0}]}))
    return aus.getvalue()


def test_format1_mit_beilage_wird_gelesen_und_geflaggt():
    p = paket.lies(_format1(True, True))
    assert p["genau"] is True
    assert p["segmente"][0]["origin"] == "machine"       # Schema 1: Maschine
    assert p["sprecher"][0]["origin"] == "human"         # benannt: Mensch


def test_rueckfall_auf_die_zeitkarte():
    """Ohne Beilage: gröber, aber MIT Wortlaut — leere Segmente wären
    schlimmer als eine Absage."""
    p = paket.lies(_format1(False, True))
    assert p["genau"] is False
    assert p["segmente"] and all(s["text"].strip() for s in p["segmente"])
    assert p["segmente"][0]["origin"] == "source"
    assert p["sprecher"]


def test_ohne_zeitkarte_und_beilage_klare_absage():
    with pytest.raises(paket.PaketFehler, match="transkript.json"):
        paket.lies(_format1(False, False))


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


def test_inventar_erkennt_veraenderte_datei(client, eintrag):
    """Format 2: jede Datei steht mit Hash im Manifest. Wird eine
    verändert, ist der Container nicht mehr lesbar — laut, nicht still."""
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    alt = zipfile.ZipFile(io.BytesIO(inhalt))
    aus = io.BytesIO()
    with zipfile.ZipFile(aus, "w", zipfile.ZIP_STORED) as neu:
        for i in alt.infolist():
            roh = alt.read(i.filename)
            if i.filename.endswith("/text/clean.json"):
                roh = roh.replace(b"Hallo", b"Hallx")
            neu.writestr(i, roh)
    with pytest.raises(paket.PaketFehler, match="clean.json ver"):
        paket.lies(aus.getvalue())
