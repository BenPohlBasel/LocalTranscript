"""Die Bibliothek — das NEUE Speicher-Modell (v2-Kernumbau).

v1 hatte VTT-Dateien als Datenbank (Sprecher = Text-Präfix, Umbenennen =
Regex über Dateien, Editor-Speichern = destruktives Überschreiben,
Neustart = alle Jobs verloren). v2: EIN Ordner je Transkript unter der
Bibliotheks-Wurzel:

    <root>/<slug>/
        transkript.json      ← kanonische Wahrheit (atomar geschrieben)
        audio.<ext>          ← Original-Audio (kopiert)
        history/<stamp>.json ← Snapshot VOR jedem Write (30 rotierend)

transkript.json (schema 1):
    {schema, id, name, created, updated,
     audio: "audio.m4a" | null,
     quelle: {datei, model, language, diarize, ...},
     sprecher: [{id, name}],            ← ENTITÄTEN; Farbe macht das UI
     segmente: [{id, start, end, sprecher: id|null, text}]}

Umbenennen/Umhängen/Zusammenführen sind damit Struktur-Operationen,
nie Text-Ersetzung. Löschen verschiebt nach _papierkorb/ (nie
destruktiv). Exporte (vtt/csv/txt/enrich) werden ABGELEITET.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import library_root

SCHEMA = 1
HISTORY_MAX = 30
AUDIO_ENDUNGEN = (".mp3", ".m4a", ".aac", ".wav", ".ogg", ".flac",
                  ".webm")


class BibliothekFehler(Exception):
    pass


def _root() -> Path:
    r = library_root()
    if r is None:
        raise BibliothekFehler("Kein Speicherort konfiguriert")
    r.mkdir(parents=True, exist_ok=True)
    return r


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _slug(name: str) -> str:
    s = re.sub(r"[^\w\- ]+", "", name).strip().replace(" ", "-")[:60]
    return s or "transkript"


def _neuer_ordner(name: str) -> Path:
    root = _root()
    s = _slug(name)
    basis, i = s, 2
    while (root / s).exists():
        s = f"{basis}-{i}"
        i += 1
    p = root / s
    p.mkdir(parents=True)
    return p


def eintrag_pfad(eid: str) -> Path:
    p = _root() / eid
    if not (p / "transkript.json").is_file():
        raise BibliothekFehler(f"Transkript nicht gefunden: {eid}")
    return p


def lese(eid: str) -> dict:
    p = eintrag_pfad(eid) / "transkript.json"
    return json.loads(p.read_text("utf-8"))


def _atomar(pfad: Path, daten: dict) -> None:
    tmp = pfad.with_suffix(".tmp")
    tmp.write_text(json.dumps(daten, ensure_ascii=False, indent=1),
                   "utf-8")
    tmp.replace(pfad)


def schreibe(eid: str, daten: dict) -> dict:
    """Snapshot des alten Stands nach history/, dann atomar schreiben."""
    ordner = eintrag_pfad(eid)
    tj = ordner / "transkript.json"
    hist = ordner / "history"
    hist.mkdir(exist_ok=True)
    shutil.copyfile(tj, hist / f"{_stamp()}.json")
    alte = sorted(hist.glob("*.json"))
    for alt in alte[:-HISTORY_MAX]:
        alt.unlink()
    daten["updated"] = datetime.now(UTC).isoformat(
        timespec="seconds")
    _atomar(tj, daten)
    return daten


def anlegen(name: str, segmente: list[dict], sprecher: list[dict],
            quelle: dict, audio: Path | None = None) -> dict:
    """Neuer Eintrag (aus Transkriptions-Job oder Import). segmente:
    [{start, end, sprecher: id|None, text}] — IDs werden hier vergeben,
    wenn sie fehlen."""
    ordner = _neuer_ordner(name)
    audio_name = None
    if audio is not None and audio.is_file():
        audio_name = f"audio{audio.suffix.lower()}"
        shutil.copyfile(audio, ordner / audio_name)
    for s in segmente:
        s.setdefault("id", uuid.uuid4().hex[:8])
        s["text"] = s.get("text", "")
    jetzt = datetime.now(UTC).isoformat(timespec="seconds")
    daten = {"schema": SCHEMA, "id": ordner.name, "name": name,
             "created": jetzt, "updated": jetzt, "audio": audio_name,
             "quelle": quelle, "sprecher": sprecher,
             "segmente": segmente}
    _atomar(ordner / "transkript.json", daten)
    return daten


def liste() -> list[dict]:
    root = library_root()
    if root is None or not root.is_dir():
        return []
    aus = []
    for tj in sorted(root.glob("*/transkript.json")):
        try:
            d = json.loads(tj.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        seg = d.get("segmente", [])
        aus.append({
            "id": d.get("id", tj.parent.name), "name": d.get("name", ""),
            "created": d.get("created", ""), "updated": d.get("updated", ""),
            "dauer": round(max((s.get("end", 0) for s in seg),
                               default=0.0), 1),
            "segmente": len(seg),
            "sprecher": len(d.get("sprecher", [])),
            "audio": bool(d.get("audio")),
            "quelle": d.get("quelle", {})})
    aus.sort(key=lambda e: e["updated"], reverse=True)
    return aus


def umbenennen(eid: str, name: str) -> dict:
    d = lese(eid)
    d["name"] = name.strip() or d["name"]
    return schreibe(eid, d)


def loeschen(eid: str) -> None:
    """In den Papierkorb der Bibliothek — nie destruktiv."""
    ordner = eintrag_pfad(eid)
    korb = _root() / "_papierkorb"
    korb.mkdir(exist_ok=True)
    ziel = korb / f"{ordner.name}-{_stamp()}"
    shutil.move(str(ordner), str(ziel))


def audio_pfad(eid: str) -> Path | None:
    d = lese(eid)
    if not d.get("audio"):
        return None
    p = eintrag_pfad(eid) / d["audio"]
    return p if p.is_file() else None


def sprecher_name(daten: dict, sid: str | None) -> str:
    if not sid:
        return ""
    for sp in daten.get("sprecher", []):
        if sp["id"] == sid:
            return sp["name"]
    return ""


def export_segmente(daten: dict) -> list[dict]:
    """Kanonisch → Export-Form (Anzeigenamen aufgelöst)."""
    return [{"start": s["start"], "end": s["end"],
             "sprecher": sprecher_name(daten, s.get("sprecher")),
             "text": s["text"]}
            for s in daten.get("segmente", [])]
