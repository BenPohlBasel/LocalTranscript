"""Speicher-Modell: kanonisches JSON, History-Snapshots, Papierkorb."""
from __future__ import annotations

import json


def test_import_und_kanonisches_modell(client, eintrag):
    d = client.get(f"/api/transcripts/{eintrag}").json()
    assert d["schema"] == 1
    assert [s["name"] for s in d["sprecher"]] == ["Anna", "Ben"]
    seg = d["segmente"]
    assert len(seg) == 2  # präfixloser Cue setzt Bens Turn fort
    sid = {s["name"]: s["id"] for s in d["sprecher"]}
    assert seg[0]["sprecher"] == sid["Anna"]
    assert seg[1]["sprecher"] == sid["Ben"]
    assert seg[1]["end"] == 11.0
    liste = client.get("/api/transcripts").json()["transcripts"]
    assert liste[0]["id"] == eintrag and liste[0]["sprecher"] == 2


def test_speichern_schreibt_history(client, eintrag):
    d = client.get(f"/api/transcripts/{eintrag}").json()
    d["segmente"][0]["text"] = "Korrigierter Wortlaut."
    r = client.put(f"/api/transcripts/{eintrag}",
                   json={"sprecher": d["sprecher"],
                         "segmente": d["segmente"]})
    assert r.status_code == 200
    neu = client.get(f"/api/transcripts/{eintrag}").json()
    assert neu["segmente"][0]["text"] == "Korrigierter Wortlaut."
    # History hält den ALTEN Stand
    from localtranscript import bibliothek
    hist = sorted((bibliothek.eintrag_pfad(eintrag) / "history")
                  .glob("*.json"))
    assert len(hist) == 1
    alt = json.loads(hist[0].read_text("utf-8"))
    assert alt["segmente"][0]["text"].startswith("Hallo")


def test_unbekannter_sprecher_422(client, eintrag):
    d = client.get(f"/api/transcripts/{eintrag}").json()
    d["segmente"][0]["sprecher"] = "sp99"
    r = client.put(f"/api/transcripts/{eintrag}",
                   json={"sprecher": d["sprecher"],
                         "segmente": d["segmente"]})
    assert r.status_code == 422


def test_umbenennen_und_papierkorb(client, eintrag):
    r = client.post(f"/api/transcripts/{eintrag}/rename",
                    json={"name": "Sitzung 1"})
    assert r.json()["name"] == "Sitzung 1"
    assert client.post(f"/api/transcripts/{eintrag}/delete",
                       json={"confirm": "falsch"}).status_code == 409
    r = client.post(f"/api/transcripts/{eintrag}/delete",
                    json={"confirm": eintrag})
    assert r.status_code == 200
    assert client.get(f"/api/transcripts/{eintrag}").status_code == 404
    # Papierkorb hält den Ordner (nie destruktiv)
    from localtranscript.config import library_root
    korb = list((library_root() / "_papierkorb").iterdir())
    assert len(korb) == 1 and korb[0].name.startswith(eintrag)


def test_csv_import(client):
    csv_daten = ('"Time-in","Time-out","Speaker","Text"\r\n'
                 '"00:00:01","00:00:05","Carla","Erster Satz."\r\n'
                 '"00:00:05","00:00:09","","Ohne Sprecher."\r\n')
    r = client.post("/api/import",
                    files={"datei": ("t.csv", csv_daten.encode(),
                                     "text/csv")})
    assert r.status_code == 200
    d = client.get(f"/api/transcripts/{r.json()['eintrag']}").json()
    assert d["segmente"][0]["sprecher"] is not None
    assert d["segmente"][1]["sprecher"] is None
