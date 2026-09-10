"""Format 2 (FORMAT.md, 2026-09-10): Herkunft je Record, Journal,
Schicht-Köpfe, Kette — aus LocalTranscripts Sicht."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import zipfile

from localtranscript import bibliothek, exporte


def _container(eintrag):
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    z = zipfile.ZipFile(io.BytesIO(inhalt))
    w = z.namelist()[0].split("/", 1)[0]
    return z, w, json.loads(z.read(f"{w}/manifest.json"))


def test_jede_schicht_traegt_den_kopf(client, eintrag):
    z, w, m = _container(eintrag)
    for pfad, f in m["files"].items():
        if f["role"] != "layer":
            continue
        d = json.loads(z.read(f"{w}/{pfad}"))
        for feld in ("kind", "id", "origin", "from", "by", "did", "result"):
            assert feld in d, (pfad, feld)
        assert d["origin"] in ("source", "machine", "llm", "human", "mixed")
        assert d["id"] == f["layer"] and m["layers"][d["id"]]["path"] == pfad


def test_inventar_ist_vollstaendig_und_stimmt(client, eintrag):
    z, w, m = _container(eintrag)
    im_zip = {n.split("/", 1)[1] for n in z.namelist()} - {"manifest.json"}
    assert set(m["files"]) == im_zip
    for pfad, f in m["files"].items():
        roh = z.read(f"{w}/{pfad}")
        assert f["hash"] == "sha256:" + hashlib.sha256(roh).hexdigest()
        assert f["bytes"] == len(roh)


def test_journal_ist_verkettet(client, eintrag):
    _z, _w, m = _container(eintrag)
    runs = m["runs"]
    assert runs and runs[0]["prev"] is None
    from enrich_core.dossier import run_eintrag_hash
    from enrich_core.schemas.manifest import RunRecord
    for vor, nach in itertools.pairwise(runs):
        assert nach["prev"] == run_eintrag_hash(RunRecord.model_validate(vor))
    eigene = [r for r in runs if r["tool"] == "localtranscript"]
    assert eigene and runs[:len(eigene)] == eigene       # Bibliothek zuerst
    for r in eigene:
        assert r["origin"] in ("source", "machine", "llm", "human")
        assert r["who"]["app"] == "localtranscript"
        assert r["who"]["install"].startswith("ins-")
        assert r["summary"]["did"] and r["started"]
        for verboten in ("did", "from", "by", "result"):     # FORMAT.md §3.1
            assert verboten not in r


def test_editor_macht_records_human_und_schreibt_ins_journal(client, eintrag):
    d = client.get(f"/api/transcripts/{eintrag}").json()
    assert all(s["origin"] == "source" for s in d["segmente"])   # Import einer VTT
    d["segmente"][0]["text"] = "Korrigiert."
    d["sprecher"][0]["name"] = "Anna Meier"
    r = client.put(f"/api/transcripts/{eintrag}",
                   json={"sprecher": d["sprecher"], "segmente": d["segmente"]})
    assert r.status_code == 200, r.text
    neu = bibliothek.lese(eintrag)
    assert neu["segmente"][0]["origin"] == "human"
    assert neu["segmente"][1]["origin"] == "source"           # unberührt bleibt
    assert neu["sprecher"][0]["origin"] == "human"
    run = neu["journal"][-1]
    assert run["origin"] == "human"
    assert run["changed"] == {d["segmente"][0]["id"]: "text",
                              d["sprecher"][0]["id"]: "name"}
    # zweites Sichern in derselben Sitzung: derselbe Run wächst
    d["segmente"][1]["text"] = "Auch korrigiert."
    client.put(f"/api/transcripts/{eintrag}",
               json={"sprecher": d["sprecher"], "segmente": d["segmente"]})
    neu2 = bibliothek.lese(eintrag)
    assert len(neu2["journal"]) == len(neu["journal"])
    assert neu2["journal"][-1]["changed"][d["segmente"][1]["id"]] == "text"


def test_herkunft_und_journal_ueberleben_den_rundlauf(client, eintrag):
    d = client.get(f"/api/transcripts/{eintrag}").json()
    d["segmente"][0]["text"] = "Von Hand."
    client.put(f"/api/transcripts/{eintrag}",
               json={"sprecher": d["sprecher"], "segmente": d["segmente"]})
    inhalt, name, _m = exporte.export_bytes(eintrag, "enrich")
    r = client.post("/api/import", files={"datei": (name, inhalt, "application/zip")})
    assert r.status_code == 200, r.text
    neu = bibliothek.lese(r.json()["eintrag"])
    assert neu["segmente"][0]["origin"] == "human"
    assert neu["segmente"][1]["origin"] == "source"
    arten = [x["origin"] for x in neu["journal"]]
    assert arten[-1] == "source" and "human" in arten     # Import-Run hinten, Historie davor


def test_schema1_wird_beim_lesen_ergaenzt(client, eintrag, tmp_path):
    p = bibliothek.eintrag_pfad(eintrag) / "transkript.json"
    alt = json.loads(p.read_text("utf-8"))
    alt["schema"] = 1
    for s in alt["segmente"]: s.pop("origin", None)
    for sp in alt["sprecher"]: sp.pop("origin", None)
    alt["sprecher"].append({"id": "spX", "name": "Sprecher 3"})
    alt.pop("journal", None)
    p.write_text(json.dumps(alt), "utf-8")
    d = bibliothek.lese(eintrag)
    assert d["schema"] == 2 and d["journal"] == []
    assert all(s["origin"] == "machine" for s in d["segmente"])
    by = {sp["name"]: sp["origin"] for sp in d["sprecher"]}
    assert by["Sprecher 3"] == "machine" and by["Anna"] == "human"


def test_enrich_liest_den_container(client, eintrag, tmp_path):
    """Der Kompatibilitätstest: enrich-core öffnet das Dossier aus dem
    Container, Inventar und Kette sind sauber, die Transkript-Schicht
    steht als deklarierte unbekannte Datei im Inventar."""
    from enrich_core.dossier import Dossier
    inhalt, name, _m = exporte.export_bytes(eintrag, "enrich")
    datei = tmp_path / name; datei.write_bytes(inhalt)
    d = Dossier.uebernehmen(datei, tmp_path / "arbeit")
    inv = d.inventar_pruefen()
    assert inv == {"fehlend": [], "undeklariert": [], "abweichend": []}, inv
    kette = d.journal_pruefen()
    assert not kette.get("kette"), kette
    m = d.manifest
    assert m.source.kind == "transcript" and m.source.canonical == "transcript.json"
    assert m.files["transcript.json"].role == "layer"
    assert any(li.kind == "transcript" and li.current for li in m.layers.values())
