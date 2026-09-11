"""Zotero-Metadaten: lokal, lesend, nur mit Einwilligung — und als
Schicht `source/zotero.json` im .enrich-Container (enrich liest sie).

Die Datenbank wird nicht gebraucht: `items_with_pdfs` und der Zugang
sind gefälscht, geprüft wird der Weg durch LocalTranscript."""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from localtranscript import bibliothek, exporte, zotero

ITEMS = [
    {"item_key": "ABCD1234", "item_type": "interview", "citekey": "whitfield2026",
     "title": "Interview Wohngenossenschaften Basel", "date": "2026-03-04",
     "year": "2026", "publication": None, "doi": None, "abstract": "x",
     "select_link": "zotero://select/library/items/ABCD1234", "library": "Meine",
     "creators": [{"first": "Nora", "last": "Whitfield", "role": "interviewer"},
                  {"first": "Julian", "last": "Marsh", "role": "interviewee"}]},
    {"item_key": "ZZZZ9999", "item_type": "book", "citekey": "marsh2019",
     "title": "Stadt und Genossenschaft", "date": "2019", "year": "2019",
     "select_link": "zotero://select/library/items/ZZZZ9999",
     "creators": [{"first": "Julian", "last": "Marsh", "role": "author"}]},
]


@pytest.fixture
def zot(monkeypatch, tmp_path):
    monkeypatch.setattr(zotero.ez, "items_with_pdfs",
                        lambda *a, **k: [dict(i) for i in ITEMS])
    monkeypatch.setattr(zotero.ez, "find_zotero_dir", lambda cfg=None: tmp_path)
    return tmp_path


def test_zotero_datum_wird_lesbar():
    assert zotero.datum("2006-04-01 2006-04-01") == "2006-04-01"
    assert zotero.datum("2019-00-00 2019") == "2019"
    assert zotero.datum("2021-03-00 März 2021") == "2021-03"
    assert zotero.datum("ca. 1990") == "ca. 1990"
    assert zotero.datum(None) is None


def test_ohne_einwilligung_424(client, zot):
    r = client.get("/api/zotero/candidates?q=Basel")
    assert r.status_code == 424
    st = client.get("/api/zotero/status").json()
    assert st["consent"] is False and st["found"] is None      # nicht gesucht


def test_kandidaten_interview_zuerst(client, zot):
    client.post("/api/settings", json={"zotero_consent": True})
    r = client.get("/api/zotero/candidates?q=Genossenschaft Basel")
    assert r.status_code == 200, r.text
    k = r.json()["candidates"]
    assert [c["item_key"] for c in k] == ["ABCD1234", "ZZZZ9999"]
    assert "abstract" not in k[0]           # die Liste bleibt schlank


def test_verknuepfen_waehlt_rollen_und_journalt(client, zot, eintrag):
    """Nur die gewählten Rollen kommen ins Transkript (Pseudonymisierung);
    der Block ist `source`, der Akt ein menschlicher Run."""
    client.post("/api/settings", json={"zotero_consent": True})
    r = client.post(f"/api/transcripts/{eintrag}/zotero",
                    json={"item_key": "ABCD1234", "roles": ["interviewer"]})
    assert r.status_code == 200, r.text
    d = bibliothek.lese(eintrag)
    z = d["zotero"]
    assert z["origin"] == "source" and z["citekey"] == "whitfield2026"
    assert [c["last"] for c in z["creators"]] == ["Whitfield"]
    assert d["journal"][-1]["origin"] == "human"
    assert d["journal"][-1]["changed"] == {"zotero": "linked"}
    # der Editor speichert weiter, ohne den Block zu kennen
    r = client.put(f"/api/transcripts/{eintrag}",
                   json={"sprecher": d["sprecher"], "segmente": d["segmente"]})
    assert r.status_code == 200, r.text
    assert bibliothek.lese(eintrag)["zotero"]["citekey"] == "whitfield2026"
    r = client.delete(f"/api/transcripts/{eintrag}/zotero")
    assert r.status_code == 200
    d = bibliothek.lese(eintrag)
    assert "zotero" not in d and d["journal"][-1]["changed"] == {"zotero": "removed"}


def test_export_traegt_zotero_schicht(client, zot, eintrag):
    client.post("/api/settings", json={"zotero_consent": True})
    client.post(f"/api/transcripts/{eintrag}/zotero",
                json={"item_key": "ABCD1234", "roles": ["interviewer", "interviewee"]})
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    z = zipfile.ZipFile(io.BytesIO(inhalt))
    w = z.namelist()[0].split("/", 1)[0]
    m = json.loads(z.read(f"{w}/manifest.json"))
    assert "source/zotero.json" in m["files"]
    zl = json.loads(z.read(f"{w}/source/zotero.json"))
    assert zl["item_key"] == "ABCD1234" and len(zl["creators"]) == 2
    # der Run steht im Journal — schlank (FORMAT.md §3.1): `layer` ist die
    # Schicht-id, die Lineage sagt, welche Datei das ist
    lay = next(k for k, v in m["layers"].items() if v["path"] == "source/zotero.json")
    assert any(r["layer"] == lay for r in m["runs"])
    # Die Kopfzeile «Titel · Datum · Interviewer:in · Citekey» setzt
    # enrich aus dieser Schicht, wenn es die Lesefassung baut — hier
    # reist nur die Schicht mit, kein PDF
    assert not any(n.endswith(".pdf") for n in z.namelist())
    # und beim Wiedereinlesen kommt der Block als Quelle zurück
    r = client.post("/api/import", files={"datei": ("x.enrich", inhalt, "application/zip")})
    assert r.status_code == 200, r.text
    neu = bibliothek.lese(r.json()["eintrag"])
    assert neu["zotero"]["citekey"] == "whitfield2026"
    assert neu["zotero"]["origin"] == "source"
    assert neu["zotero"]["select_link"].startswith("zotero://select/")


def test_fremder_select_link_wird_verworfen(client, zot, eintrag):
    """Ein Dossier von anderswo darf der App keinen Link unterschieben,
    den «In Zotero zeigen» dann an `open` gibt (Review 2026-09-11)."""
    from enrich_core.canonical import content_hash
    client.post("/api/settings", json={"zotero_consent": True})
    client.post(f"/api/transcripts/{eintrag}/zotero",
                json={"item_key": "ABCD1234", "roles": ["interviewer"]})
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    alt = zipfile.ZipFile(io.BytesIO(inhalt)); w = alt.namelist()[0].split("/", 1)[0]
    m = json.loads(alt.read(f"{w}/manifest.json"))
    zl = json.loads(alt.read(f"{w}/source/zotero.json"))
    zl["select_link"] = "https://evil.example/t?id=1"
    roh = json.dumps(zl).encode()
    m["files"]["source/zotero.json"]["hash"] = content_hash(roh)
    m["files"]["source/zotero.json"]["bytes"] = len(roh)
    aus = io.BytesIO()
    with zipfile.ZipFile(aus, "w", zipfile.ZIP_STORED) as neu:
        for i in alt.infolist():
            if i.filename.endswith("/source/zotero.json"): neu.writestr(i, roh)
            elif i.filename.endswith("/manifest.json"): neu.writestr(i, json.dumps(m).encode())
            else: neu.writestr(i, alt.read(i.filename))
    r = client.post("/api/import", files={"datei": ("x.enrich", aus.getvalue(), "application/zip")})
    assert r.status_code == 200, r.text
    assert bibliothek.lese(r.json()["eintrag"])["zotero"]["select_link"] is None
