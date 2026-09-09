"""REFI-QDA-Export: Form gegen eine ECHTE ATLAS.ti-Datei geprüft.

Die Referenzwerte stammen aus einem Export von ATLAS.ti 25.0.1 (macOS,
Audio-Transkript, User 2026-09-09): Medien liegen über
`relative:///` AUSSEN in `<Name> Media/`, der Plain Text hat KEIN BOM,
und jeder SyncPoint sitzt auf einem Zeilenanfang (dort 2805 von 2805).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from localtranscript import qdpx

NS = "{urn:QDA-XML:project:1.0}"

#: Audio-Platzhalter — für die FORM des Pakets zählt nur, dass die
#: Datei existiert; der Inhalt wandert unverändert ins Zip.
AUDIO = Path(__file__)


def _paket(inhalt: bytes) -> tuple[ET.Element, str, list[str]]:
    """Äußeres Zip → (project.qde-Wurzel, Plain Text, Dateiliste)."""
    aussen = zipfile.ZipFile(io.BytesIO(inhalt))
    qdpx_name = next(n for n in aussen.namelist() if n.endswith(".qdpx"))
    innen = zipfile.ZipFile(io.BytesIO(aussen.read(qdpx_name)))
    qde = next(n for n in innen.namelist() if n.endswith(".qde"))
    txt = next(n for n in innen.namelist() if n.endswith(".txt"))
    return (ET.fromstring(innen.read(qde)),
            innen.read(txt).decode("utf-8"),
            aussen.namelist())


#: Export-Form: `sprecher` ist der Anzeigename (bibliothek.export_segmente)
SEGMENTE = [
    {"start": 0.0, "end": 2.5, "sprecher": "Anna",
     "text": "Erste Zeile."},
    {"start": 2.5, "end": 5.0, "sprecher": "Anna",
     "text": "Noch Anna, ohne neues Präfix."},
    {"start": 5.0, "end": 7.25, "sprecher": "Ben",
     "text": "Jetzt Ben."},
    {"start": 7.25, "end": 9.0, "sprecher": None,
     "text": "Ohne Sprecher."},
]
SPRECHER = [{"id": "a", "name": "Anna"}, {"id": "b", "name": "Ben"}]


def test_text_praefix_nur_beim_wechsel():
    text, marken = qdpx.text_und_marken(SEGMENTE)
    zeilen = text.splitlines()
    assert zeilen == ["Anna: Erste Zeile.",
                      "Noch Anna, ohne neues Präfix.",
                      "Ben: Jetzt Ben.",
                      "Ohne Sprecher."]
    # jede Marke zeigt auf den Anfang ihrer Zeile
    for m, zeile in zip(marken, zeilen, strict=True):
        assert text[m["pos"]:m["pos"] + len(zeile)] == zeile
    assert [m["ms"] for m in marken] == [0, 2500, 5000, 7250]


def test_leere_segmente_fallen_weg():
    text, marken = qdpx.text_und_marken(
        [{"start": 0.0, "end": 1.0, "sprecher": "Anna", "text": "   "},
         {"start": 1.0, "end": 2.0, "sprecher": "Anna", "text": "Da."}])
    assert text == "Anna: Da.\n"
    assert len(marken) == 1 and marken[0]["pos"] == 0


def test_syncpoints_auf_zeilenanfang_und_monoton():
    daten = qdpx.baue_zip("Probe", SEGMENTE, SPRECHER, None)
    wurzel, text, _ = _paket(daten)
    # ohne Audio gibt es keine AudioSource — und damit keine SyncPoints
    assert wurzel.find(f".//{NS}AudioSource") is None

    daten = qdpx.baue_zip("Probe", SEGMENTE, SPRECHER, AUDIO)
    wurzel, text, _dateien = _paket(daten)
    punkte = [(int(e.get("position")), int(e.get("timeStamp")))
              for e in wurzel.iter(f"{NS}SyncPoint")]
    assert len(punkte) == len(SEGMENTE)
    anfaenge = {0} | {i + 1 for i, c in enumerate(text) if c == "\n"}
    assert all(p in anfaenge for p, _ in punkte)
    assert punkte == sorted(punkte)
    assert punkte[0] == (0, 0)


def test_medien_liegen_aussen_wie_bei_atlas():
    daten = qdpx.baue_zip("Mein Projekt", SEGMENTE, SPRECHER, AUDIO)
    wurzel, _text, dateien = _paket(daten)
    assert "Mein Projekt.qdpx" in dateien
    assert any(n.startswith("Mein Projekt Media/") for n in dateien)
    quelle = wurzel.find(f".//{NS}AudioSource")
    assert quelle.get("path").startswith("relative:///")
    # der Dateiname im Zip ist genau der aus dem path
    ziel = quelle.get("path")[len("relative:///"):]
    assert f"Mein Projekt Media/{ziel}" in dateien


def test_plain_text_ohne_bom():
    daten = qdpx.baue_zip("Probe", SEGMENTE, SPRECHER, None)
    aussen = zipfile.ZipFile(io.BytesIO(daten))
    innen = zipfile.ZipFile(io.BytesIO(aussen.read("Probe.qdpx")))
    roh = innen.read(next(n for n in innen.namelist()
                          if n.endswith(".txt")))
    assert not roh.startswith(b"\xef\xbb\xbf")


def test_sprecher_werden_codes_und_selektionen():
    daten = qdpx.baue_zip("Probe", SEGMENTE, SPRECHER, None)
    wurzel, text, _ = _paket(daten)
    codes = {e.get("guid"): e.get("name")
             for e in wurzel.iter(f"{NS}Code")}
    assert sorted(codes.values()) == ["Anna", "Ben"]
    sel = list(wurzel.iter(f"{NS}PlainTextSelection"))
    # drei Segmente mit Sprecher, das vierte ohne bleibt uncodiert
    assert len(sel) == 3
    for e in sel:
        a, b = int(e.get("startPosition")), int(e.get("endPosition"))
        ziel = e.find(f".//{NS}CodeRef").get("targetGUID")
        name = codes[ziel]
        zeile = text[a:b]
        assert zeile.startswith(f"{name}: ") or ": " not in zeile


def test_guids_sind_deterministisch():
    a = qdpx.baue_projekt("Probe", SEGMENTE, SPRECHER, "x.mp3")[0]
    b = qdpx.baue_projekt("Probe", SEGMENTE, SPRECHER, "x.mp3")[0]
    # Zeitstempel unterscheiden sich ggf. — GUIDs nie
    guids_a = sorted(e.get("guid") for e in ET.fromstring(a).iter()
                     if e.get("guid"))
    guids_b = sorted(e.get("guid") for e in ET.fromstring(b).iter()
                     if e.get("guid"))
    assert guids_a == guids_b
    assert all(g == g.upper() for g in guids_a)


def test_endpunkt_liefert_paket(client, eintrag):
    r = client.get(f"/api/transcripts/{eintrag}/export/qdpx")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    wurzel, text, _ = _paket(r.content)
    assert wurzel.tag == f"{NS}Project"
    assert wurzel.get("origin", "").startswith("LocalTranscript")
    assert "Anna: " in text and "Ben: " in text
