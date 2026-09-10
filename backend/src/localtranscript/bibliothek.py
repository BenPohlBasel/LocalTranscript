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

from .config import identitaet, library_root, ulid

#: Schema 2 (2026-09-10, FORMAT.md §0.1/§3.1): jedes Segment und jeder
#: Sprecher trägt `origin` (source | machine | llm | human), die Datei
#: ein `journal` — wer hat wann was geschrieben. Schema 1 wird beim
#: Lesen ergänzt (§ _ergaenze_schema1) und beim nächsten Schreiben
#: festgeschrieben.
SCHEMA = 2
ORIGINS = ("source", "machine", "llm", "human")
#: Standardnamen der Diarisierung — solche Sprecher sind `machine`,
#: benannte sind `human` (Schema-1-Migration).
_STANDARDNAME = re.compile(r"^(Sprecher|Speaker|Locuteur|Parlante) \d+$")
#: Editor-Sitzung: Schreibungen innerhalb dieser Ruhezeit bilden EINEN
#: Journal-Eintrag (FORMAT.md §3.1 — je Sitzung, nicht je Tastendruck)
SITZUNG_RUHE_S = 600
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
    return _ergaenze_schema1(json.loads(p.read_text("utf-8")))


def _ergaenze_schema1(d: dict) -> dict:
    """Schema 1 → 2 im Speicher: Segmente ohne Flag stammen von der
    Maschine, Sprecher mit Standardnamen auch, benannte von Menschen.
    Ein leeres Journal bleibt leer — nichts erfinden."""
    if d.get("schema", 1) >= SCHEMA:
        return d
    for seg in d.get("segmente", []):
        seg.setdefault("origin", "machine")
    for sp in d.get("sprecher", []):
        sp.setdefault("origin", "machine" if _STANDARDNAME.match(
            sp.get("name", "")) else "human")
    d.setdefault("journal", [])
    d["schema"] = SCHEMA
    return d


def kanonisch(daten: dict) -> str:
    """Die EINE JSON-Form (FORMAT.md §2): sortierte Schlüssel, UTF-8,
    Zeilenende — nur so sind Hashes stabil."""
    return json.dumps(daten, ensure_ascii=False, sort_keys=True,
                      indent=1) + "\n"


def _atomar(pfad: Path, daten: dict) -> None:
    tmp = pfad.with_suffix(".tmp")
    tmp.write_text(kanonisch(daten), "utf-8")
    tmp.replace(pfad)


def _jetzt() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _run_hash(run: dict) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(
        kanonisch(run).encode("utf-8")).hexdigest()


def journal_eintrag(daten: dict, *, origin: str, did: str,
                    changed: dict | None = None, layer: str = "transcript",
                    by: dict | None = None, result: dict | None = None,
                    started: str | None = None) -> dict:
    """Einen Run ans Journal hängen — klein (IDs und Hashes, nie
    Inhalte), verkettet über den Hash des vorangehenden Eintrags.
    Menschliche Runs derselben Installation innerhalb der Ruhezeit
    werden zu EINEM Eintrag zusammengefasst."""
    assert origin in ORIGINS, origin
    journal = daten.setdefault("journal", [])
    jetzt = _jetzt()
    wer = identitaet()
    if origin == "human" and journal:
        letzter = journal[-1]
        if (letzter.get("origin") == "human"
                and letzter.get("who", {}).get("install") == wer["install"]
                and _sekunden_seit(letzter.get("finished")) < SITZUNG_RUHE_S):
            letzter["finished"] = jetzt
            letzter.setdefault("changed", {}).update(changed or {})
            letzter["result"] = {"records": len(letzter["changed"])}
            return letzter
    run = {"id": f"run-{ulid()}",
           "prev": _run_hash(journal[-1]) if journal else None,
           "origin": origin, "who": dict(wer, **(by or {})),
           "started": started or jetzt, "finished": jetzt,
           "layer": layer, "did": did}
    if changed:
        run["changed"] = dict(changed)
    run["result"] = result or {"records": len(changed or {})}
    journal.append(run)
    return run


def _sekunden_seit(iso: str | None) -> float:
    if not iso:
        return float("inf")
    try:
        t = datetime.fromisoformat(iso)
    except ValueError:
        return float("inf")
    return (datetime.now(UTC) - t).total_seconds()


def _aenderungen(alt: dict, neu: dict) -> dict:
    """Welche Records hat der Editor angefasst? IDs → Art der Änderung.
    Das ist der `changed`-Block des Journals (FORMAT.md §3.1)."""
    aus: dict[str, str] = {}
    alte_seg = {s["id"]: s for s in alt.get("segmente", [])}
    for s in neu.get("segmente", []):
        a = alte_seg.get(s["id"])
        if a is None:
            aus[s["id"]] = "new"
        elif a.get("text") != s.get("text"):
            aus[s["id"]] = "text"
        elif a.get("sprecher") != s.get("sprecher"):
            aus[s["id"]] = "speaker"
        elif (a.get("start"), a.get("end")) != (s.get("start"), s.get("end")):
            aus[s["id"]] = "time"
    neue_ids = {s["id"] for s in neu.get("segmente", [])}
    for sid in alte_seg:
        if sid not in neue_ids:
            aus[sid] = "removed"
    alte_sp = {p["id"]: p for p in alt.get("sprecher", [])}
    for p in neu.get("sprecher", []):
        a = alte_sp.get(p["id"])
        if a is None:
            aus[p["id"]] = "new"
        elif a.get("name") != p.get("name"):
            aus[p["id"]] = "name"
    for pid in alte_sp:
        if pid not in {p["id"] for p in neu.get("sprecher", [])}:
            aus[pid] = "removed"
    return aus


def schreibe(eid: str, daten: dict, *, did: str = "Im Editor bearbeitet",
             origin: str = "human") -> dict:
    """Snapshot des alten Stands nach history/, dann atomar schreiben.
    Was sich gegenüber dem alten Stand geändert hat, wird `origin:
    human` (der Editor ist der einzige Weg hierher) und steht als Run
    im Journal — je Sitzung gebündelt."""
    ordner = eintrag_pfad(eid)
    tj = ordner / "transkript.json"
    hist = ordner / "history"
    hist.mkdir(exist_ok=True)
    shutil.copyfile(tj, hist / f"{_stamp()}.json")
    alte = sorted(hist.glob("*.json"))
    for alt in alte[:-HISTORY_MAX]:
        alt.unlink()
    alt_stand = _ergaenze_schema1(json.loads(tj.read_text("utf-8")))
    daten = _ergaenze_schema1(daten)
    daten["journal"] = alt_stand.get("journal", [])   # das Journal führt die Bibliothek
    changed = _aenderungen(alt_stand, daten)
    if changed:
        by_id = {s["id"]: s for s in daten["segmente"]}
        by_id.update({p["id"]: p for p in daten["sprecher"]})
        for rid, art in changed.items():
            if art != "removed" and rid in by_id:
                by_id[rid]["origin"] = origin
        journal_eintrag(daten, origin=origin, did=did, changed=changed)
    daten["updated"] = _jetzt()
    _atomar(tj, daten)
    return daten


def anlegen(name: str, segmente: list[dict], sprecher: list[dict],
            quelle: dict, audio: Path | None = None, *,
            origin: str = "machine", journal: list[dict] | None = None,
            by: dict | None = None) -> dict:
    """Neuer Eintrag (aus Transkriptions-Job oder Import). segmente:
    [{start, end, sprecher: id|None, text}] — IDs werden hier vergeben,
    wenn sie fehlen. `origin` gilt für alle Records ohne eigenes Flag:
    `machine` für Whisper, `source` für eine fremde VTT/CSV (die Datei
    ist die Quelle, wie sie kam). Ein mitgebrachtes Journal (Import aus
    einem Dossier) wird übernommen und um den Import-Run verlängert."""
    assert origin in ORIGINS, origin
    ordner = _neuer_ordner(name)
    audio_name = None
    if audio is not None and audio.is_file():
        audio_name = f"audio{audio.suffix.lower()}"
        shutil.copyfile(audio, ordner / audio_name)
    for s in segmente:
        s.setdefault("id", uuid.uuid4().hex[:8])
        s["text"] = s.get("text", "")
        s.setdefault("origin", origin)
    for sp in sprecher:
        sp.setdefault("origin", "machine" if _STANDARDNAME.match(
            sp.get("name", "")) else ("human" if origin == "machine" else origin))
    jetzt = _jetzt()
    daten = {"schema": SCHEMA, "id": ordner.name, "name": name,
             "created": jetzt, "updated": jetzt, "audio": audio_name,
             "quelle": quelle, "sprecher": sprecher,
             "segmente": segmente, "journal": list(journal or [])}
    art = quelle.get("erzeugt", "")
    did = ("Whisper-Transkript mit Sprechertrennung" if art == "transcription"
           else f"Import aus {quelle.get('datei', '?')}")
    journal_eintrag(daten, origin=origin if art != "transcription" else "machine",
                    did=did, by=by,
                    result={"records": len(segmente) + len(sprecher)})
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
