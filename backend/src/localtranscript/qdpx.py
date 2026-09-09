"""REFI-QDA-Export (.qdpx) — Transkript + Audio für ATLAS.ti & Co.

Gebaut nach einer ECHTEN Referenzdatei statt nach der Spezifikation
allein: ATLAS.ti 25.0.1 (macOS), Export eines Audio-Transkripts
(User 2026-09-09). Was daran gemessen wurde und hier nachgebaut wird:

- Auslieferform ist ein Ordner, kein einzelnes File: `<Name>.qdpx`
  (ZIP mit project.qde + sources/) und daneben `<Name> Media/` mit
  dem Medium. Wir packen beides in EIN Zip — genau wie das Archiv,
  das ATLAS.ti abliefert.
- `<AudioSource path="relative:///<guid>.mp3">` — Medien liegen
  AUSSEN. `internal://` gilt nur für Text und docx.
- `<Transcript>` in der AudioSource, darin ein `<SyncPoint>` je
  Äußerung: `position` = ZEICHEN-Offset im Plain Text (kein BOM in
  der Referenz!), `timeStamp` = Millisekunden. In der Referenz lagen
  2805 von 2805 SyncPoints exakt auf einem Zeilenanfang.
- Dasselbe Transkript hängt ZWEIMAL im Projekt: als eigenständige
  TextSource und als Transcript der AudioSource, beide auf denselben
  plainTextPath. Kein Widerspruch — so bietet ATLAS.ti es als
  Dokument UND als Medien-Spur an.
- Der Text ist `Sprecher: Wortlaut`, eine Äußerung je Zeile, das
  Präfix nur beim Sprecherwechsel. Das ist unser build_txt().

GUIDs deterministisch aus uuid5 (enrich-Muster, refi_projekt.py):
kein Zustand nötig, derselbe Export ergibt dieselben GUIDs — ein
zweiter Import überschreibt statt zu verdoppeln.

`basePath` schreiben wir NICHT: ATLAS.ti trägt dort den absoluten
Pfad seines eigenen Export-Laufs ein, der auf einer fremden Maschine
ins Leere zeigt. Ohne die Angabe bleibt dem Leser der naheliegende
Weg — neben dem Projekt suchen — und der Medien-Ordner heißt genau
so wie bei ATLAS.ti.
"""
from __future__ import annotations

import io
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from . import config

_NS = "urn:QDA-XML:project:1.0"
_XSI = "http://www.w3.org/2001/XMLSchema-instance"

#: Eigene Namensräume je Sorte — GUIDs kollidieren so nie über Sorten
#: hinweg, auch wenn zwei Objekte dieselbe id tragen.
_NS_QUELLE = uuid.uuid5(uuid.NAMESPACE_URL, "localtranscript:refi-source")
_NS_CODE = uuid.uuid5(uuid.NAMESPACE_URL, "localtranscript:refi-code")
_NS_SEL = uuid.uuid5(uuid.NAMESPACE_URL, "localtranscript:refi-selection")
_NS_USER = uuid.uuid5(uuid.NAMESPACE_URL, "localtranscript:refi-user")

#: Sprecher-Palette in der Reihenfolge des Frontends
#: (lib/api.ts SPRECHER_FARBEN) — die Codes tragen im QDA-Programm
#: dieselben Farben wie die Sprecher im Editor.
FARBEN = ("#3E63DD", "#FFB224", "#30A46C", "#E5484D", "#12A594",
          "#8E4EC6", "#F76B15", "#00A2C7", "#D6409F", "#99D52A")


def _guid(ns: uuid.UUID, schluessel: str) -> str:
    return str(uuid.uuid5(ns, schluessel)).upper()


def _hms(sekunden: float) -> str:
    s = max(0, int(sekunden))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def text_und_marken(segmente: list[dict]) -> tuple[str, list[dict]]:
    """Plain Text + je Äußerung eine Marke {pos, ms, start, end, sprecher}.

    Erwartet die EXPORT-Form (bibliothek.export_segmente): `sprecher`
    ist der Anzeigename, nicht die Entitäts-id — wie bei build_txt und
    build_csv.

    Eine Zeile je Segment (nicht je Turn wie build_txt): nur so bekommt
    jede Äußerung ihren eigenen SyncPoint. Das Sprecher-Präfix steht
    beim WECHSEL, Fortsetzungen laufen ohne — wie in der Referenz.
    """
    zeilen: list[str] = []
    marken: list[dict] = []
    offset = 0
    letzter: str | None = None
    for s in segmente:
        wortlaut = " ".join((s.get("text") or "").split())
        if not wortlaut:
            continue
        wer = s.get("sprecher") or ""
        zeile = f"{wer}: {wortlaut}" if wer and wer != letzter else wortlaut
        letzter = wer
        marken.append({"pos": offset,
                       "ms": round(float(s["start"]) * 1000),
                       "start": float(s["start"]),
                       "end": float(s["end"]),
                       "sprecher": s.get("sprecher") or "",
                       "laenge": len(zeile)})
        zeilen.append(zeile)
        offset += len(zeile) + 1          # + "\n"
    return ("\n".join(zeilen) + "\n" if zeilen else ""), marken


def baue_projekt(name: str, segmente: list[dict], sprecher: list[dict],
                 audio_name: str | None,
                 codes: bool = True) -> tuple[bytes, str, list[dict]]:
    """project.qde bauen. Gibt (XML, Plain Text, Marken) zurück.

    `segmente` in Export-Form (Namen aufgelöst), `sprecher` die
    Entitäten — sie geben Reihenfolge und damit die Farbe.
    """
    text, marken = text_und_marken(segmente)

    jetzt = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    txt_guid = _guid(_NS_QUELLE, f"{name}:text")
    audio_guid = _guid(_NS_QUELLE, f"{name}:audio")
    user_guid = _guid(_NS_USER, "LocalTranscript")

    ET.register_namespace("", _NS)
    ET.register_namespace("xsi", _XSI)
    wurzel = ET.Element("Project", {
        "xmlns": _NS,
        "xmlns:xsi": _XSI,
        "xsi:schemaLocation": f"{_NS} http://schema.qdasoftware.org"
                              "/versions/Project/v1.0/Project.xsd",
        "name": name,
        "origin": f"{config.APP_NAME} {config.APP_VERSION}",
        "creatingUserGUID": user_guid,
        "creationDateTime": jetzt,
        "modifiedDateTime": jetzt,
    })

    users = ET.SubElement(wurzel, "Users")
    ET.SubElement(users, "User", {"guid": user_guid,
                                  "name": config.APP_NAME})

    # ---------- CodeBook: ein Code je Sprecher ----------
    # Schlüssel ist der NAME (so stehen die Sprecher in den Segmenten),
    # die GUID kommt aus der stabilen Entitäts-id. Zwei gleichnamige
    # Sprecher teilen sich einen Code — im QDA-Programm ist derselbe
    # Name derselbe Mensch.
    code_guid: dict[str, str] = {}
    if codes and sprecher:
        cb = ET.SubElement(wurzel, "CodeBook")
        liste = ET.SubElement(cb, "Codes")
        for i, sp in enumerate(sprecher):
            if sp["name"] in code_guid:
                continue
            g = _guid(_NS_CODE, sp["id"])
            code_guid[sp["name"]] = g
            ET.SubElement(liste, "Code", {
                "guid": g, "name": sp["name"], "isCodable": "true",
                "color": FARBEN[i % len(FARBEN)]})

    quellen = ET.SubElement(wurzel, "Sources")

    # ---------- TextSource: das Transkript als Dokument ----------
    txt_quelle = ET.SubElement(quellen, "TextSource", {
        "guid": txt_guid, "name": name,
        "plainTextPath": f"internal://{txt_guid}.txt",
        "creatingUser": user_guid, "creationDateTime": jetzt,
        "modifyingUser": user_guid, "modifiedDateTime": jetzt})
    if code_guid:
        for m in marken:
            g = code_guid.get(m["sprecher"])
            if not g:
                continue
            sel = ET.SubElement(txt_quelle, "PlainTextSelection", {
                "guid": _guid(_NS_SEL, f"{txt_guid}:{m['pos']}"),
                "name": f"{_hms(m['start'])} – {_hms(m['end'])}",
                "startPosition": str(m["pos"]),
                "endPosition": str(m["pos"] + m["laenge"]),
                "creatingUser": user_guid, "creationDateTime": jetzt,
                "modifyingUser": user_guid, "modifiedDateTime": jetzt})
            cod = ET.SubElement(sel, "Coding", {
                "guid": _guid(_NS_SEL, f"c:{txt_guid}:{m['pos']}"),
                "creatingUser": user_guid, "creationDateTime": jetzt})
            ET.SubElement(cod, "CodeRef", {"targetGUID": g})

    # ---------- AudioSource + Transcript: dieselbe Textdatei ----------
    if audio_name:
        audio = ET.SubElement(quellen, "AudioSource", {
            "guid": audio_guid, "name": name,
            "path": f"relative:///{audio_guid}{Path(audio_name).suffix}",
            "creatingUser": user_guid, "creationDateTime": jetzt,
            "modifyingUser": user_guid, "modifiedDateTime": jetzt})
        tr = ET.SubElement(audio, "Transcript", {
            "guid": _guid(_NS_QUELLE, f"{name}:transcript"), "name": name,
            "plainTextPath": f"internal://{txt_guid}.txt",
            "creatingUser": user_guid, "creationDateTime": jetzt,
            "modifyingUser": user_guid, "modifiedDateTime": jetzt})
        for m in marken:
            ET.SubElement(tr, "SyncPoint", {
                "guid": _guid(_NS_SEL, f"sp:{m['pos']}"),
                "position": str(m["pos"]), "timeStamp": str(m["ms"])})

    ET.indent(wurzel)
    xml = ET.tostring(wurzel, encoding="utf-8", xml_declaration=True)
    return xml, text, marken


def baue_zip(name: str, segmente: list[dict], sprecher: list[dict],
             audio: Path | None, codes: bool = True) -> bytes:
    """Das ganze Paket: <Name>.qdpx + <Name> Media/ in EINEM Zip."""
    audio_name = audio.name if audio and audio.is_file() else None
    xml, text, _marken = baue_projekt(name, segmente, sprecher,
                                      audio_name, codes)
    txt_guid = _guid(_NS_QUELLE, f"{name}:text")
    audio_guid = _guid(_NS_QUELLE, f"{name}:audio")

    # inneres Zip = das .qdpx
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}.qde", xml)
        # OHNE BOM — die Referenz hat keines, und `position` zählt
        # Zeichen des BOM-losen Textes
        z.writestr(f"sources/{txt_guid}.txt", text.encode("utf-8"))

    aussen = io.BytesIO()
    # Audio wird nicht noch einmal komprimiert (mp3 ist es schon) —
    # ZIP_STORED spart bei 300-MB-Mitschnitten Minuten.
    with zipfile.ZipFile(aussen, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}.qdpx", inner.getvalue())
        if audio_name:
            z.write(audio, f"{name} Media/{audio_guid}"
                           f"{Path(audio_name).suffix}",
                    compress_type=zipfile.ZIP_STORED)
    return aussen.getvalue()
