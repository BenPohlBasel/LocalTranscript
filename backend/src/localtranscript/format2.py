"""Der .enrich-Container nach FORMAT.md, Format 2 — enrich-core schreibt,
LocalTranscript ergänzt.

Das gevendorte enrich-Werkzeug (`enrich_export.textsatz`) setzt aus dem
Transkript das Dossier: PDF, Layout, T0, T1, Zeitkarte — und enrich-core
führt dabei das Manifest nach Format 2 (Inventar `files`, Lineage
`layers`, `source`, `producer`, Journal mit `prev`-Kette). Dieser Modul
legt darauf NUR, was enrich nicht wissen kann (Stand enrich@08a0e78,
2026-09-11):

- `transcript.json` — die Quelle: das kanonische Transkript mit
  `origin` je Segment und Sprecher, Kopf nach dem Schicht-Vertrag (§2),
  als Schicht registriert (Inventar + Lineage) und in `source.canonical`
  eingetragen. Das gesetzte PDF bleibt `rendered`.
- Das Journal der Bibliothek — Whisper-Lauf, Import, Editor-Sitzungen —
  als `RunRecord`s VOR den Läufen des Textsatzes (das Transkript ist
  die Quelle aller anderen Schichten); die Kette wird über alle
  Einträge neu geschlossen, mit enrichs eigener Hash-Funktion.
- `producer` = LocalTranscript, `profile` = handover, `title`.

Alles über die Modelle von enrich-core, damit das Manifest von enrichs
Leser validiert wird — der Kompatibilitätstest öffnet den Container mit
`Dossier.open` und prüft Inventar und Kette (tests/test_format2.py).
Dateinamen bleiben, wie enrich sie heute schreibt (Nummern, Wurzel);
Anhang A der FORMAT.md ist enrichs Schritt, nicht unserer.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from enrich_core.canonical import content_hash
from enrich_core.dossier import Dossier, run_eintrag_hash
from enrich_core.ids import new_id
from enrich_core.schemas.common import Agent
from enrich_core.schemas.manifest import (
    FileEntry,
    LayerInfo,
    Producer,
    RunRecord,
    SourceInfo,
    Who,
)

from . import bibliothek
from .config import APP_VERSION, identitaet

TRANSCRIPT_DATEI = "transcript.json"
TRANSCRIPT_SCHEMA = "transcript/1.0.0"
TOOL = "localtranscript"

#: Bibliotheks-Herkunft → enrich `agent.type` (Format 1, weiter Pflicht)
_AGENT_TYPE = {"source": "extracted", "machine": "derived",
               "llm": "llm", "human": "human"}


def _kanon(d: dict) -> bytes:
    return bibliothek.kanonisch(d).encode("utf-8")


# ---------- Schreiben ----------

def transkript_schicht(daten: dict, layer_id: str, wer: dict) -> dict:
    """`transcript.json`: Kopf nach FORMAT.md §2 (in der Form, die enrichs
    LayerHead liest) plus Sprecher und Segmente mit `origin`."""
    daten = bibliothek._ergaenze_schema1(dict(daten))
    origins = {s.get("origin") for s in daten["segmente"]} | \
              {p.get("origin") for p in daten["sprecher"]}
    quelle = daten.get("quelle", {})
    by = {"tool": TOOL, "version": APP_VERSION, "model": None,
          "prompt": None, "user": wer.get("user"),
          "config": {k: quelle[k] for k in ("model", "language", "diarize")
                     if k in quelle}}
    if quelle.get("model"):
        by["model"] = f"whisper {quelle['model']}"
    return {
        "schema": TRANSCRIPT_SCHEMA, "kind": "transcript", "id": layer_id,
        "origin": next(iter(origins)) if len(origins) == 1 else "mixed",
        "from": {}, "by": by, "did": _did(daten),
        "result": {"records": len(daten["segmente"]) + len(daten["sprecher"]),
                   "relations": None, "warnings": []},
        "name": daten.get("name", ""),
        "language": quelle.get("language"),
        "speakers": [{"id": p["id"], "name": p["name"],
                      "origin": p.get("origin", "machine")}
                     for p in daten["sprecher"]],
        "segments": [{"id": s["id"], "t0_s": s["start"], "t1_s": s["end"],
                      "speaker": s.get("sprecher"), "text": s["text"],
                      "origin": s.get("origin", "machine")}
                     for s in daten["segmente"]],
    }


def _did(daten: dict) -> str:
    art = daten.get("quelle", {}).get("erzeugt", "")
    korrigiert = any(s.get("origin") == "human" for s in daten["segmente"])
    kern = ("Whisper-Transkript mit Sprechertrennung" if art == "transcription"
            else f"Importiert aus {daten.get('quelle', {}).get('datei', '?')}")
    return kern + (", danach von Hand korrigiert." if korrigiert else ".")


def _run_aus_journal(r: dict, wer: dict) -> RunRecord:
    """Bibliotheks-Run → enrich RunRecord. Nur die vier Fragen (wann,
    was, wer, wo); `did` steht als einziger Satz in `summary`."""
    origin = r.get("origin", "machine")
    who = dict(r.get("who") or {})
    return RunRecord(
        id=r.get("id") or new_id("run"), tool=TOOL, tool_version=APP_VERSION,
        layer=TRANSCRIPT_DATEI,
        # enrichs Agent (Format 1) verlangt bei human eine Adresse; FORMAT.md
        # §3.1 sagt «unbekannte Person ist kein Fehler». Bis enrich das
        # angleicht: die Installations-Kennung als stabile Adresse —
        # unterscheidbar, nicht rückführbar (wie in `who`).
        agent=Agent(type=_AGENT_TYPE.get(origin, "derived"),
                    tool=f"{TOOL}/{APP_VERSION}",
                    model=who.get("model"),
                    user=who.get("user") or (
                        who.get("install", wer.get("install"))
                        if origin == "human" else None)),
        started=r.get("started") or bibliothek._jetzt(),
        finished=r.get("finished"),
        summary={"did": r.get("did", "")},
        origin=origin,
        # Bibliothek führt `app` als «localtranscript/2.2.0»; enrichs Who
        # trennt app und version
        who=Who(app=TOOL, version=(who.get("app") or f"{TOOL}/{APP_VERSION}").split("/", 1)[-1],
                install=who.get("install", wer.get("install")),
                user=who.get("user")),
        changed=dict(r.get("changed") or {}),
    )


def baue_container(daten: dict, d: Dossier, stamm: str) -> bytes:
    """Dossier (von textsatz gebaut) + kanonisches Transkript → Container."""
    wer = identitaet()
    layer_id = new_id("lay")
    schicht = _kanon(transkript_schicht(daten, layer_id, wer))
    (d.path / TRANSCRIPT_DATEI).write_bytes(schicht)

    m = d.manifest
    # Journal: Bibliothek zuerst, dann der Textsatz; Kette neu schliessen
    eigene = [_run_aus_journal(r, wer) for r in daten.get("journal", [])]
    alle = eigene + list(m.runs)
    vorher: RunRecord | None = None
    for run in alle:
        run.prev = run_eintrag_hash(vorher) if vorher is not None else None
        vorher = run
    m.runs = alle
    m.layers[layer_id] = LayerInfo(
        path=TRANSCRIPT_DATEI, kind="transcript", origin=schicht_origin(schicht),
        current=True, hash=content_hash(schicht),
        run=eigene[-1].id if eigene else None,
        **{"schema": TRANSCRIPT_SCHEMA, "from": {}})
    m.files[TRANSCRIPT_DATEI] = FileEntry(
        role="layer", hash=content_hash(schicht), bytes=len(schicht),
        layer=layer_id)
    alt = m.source
    m.source = SourceInfo(kind="transcript", canonical=TRANSCRIPT_DATEI,
                          media=alt.media if alt else None,
                          rendered=alt.rendered if alt else "source.pdf")
    # enrich trägt source.pdf beim Anlegen als `source` ein, bevor es
    # weiss, dass die Quelle ein Transkript ist — für uns ist das PDF die
    # Lesefassung (FORMAT.md §3), aus dem Transkript gesetzt.
    pdf = m.files.get("source.pdf")
    if pdf is not None:
        m.files["source.pdf"] = FileEntry(role="rendered", hash=pdf.hash,
                                          bytes=pdf.bytes, layer=None,
                                          **{"from": TRANSCRIPT_DATEI})
    m.producer = Producer(app=TOOL, version=APP_VERSION)
    m.profile = "handover"
    m.title = daten.get("name") or stamm
    d._write_manifest(m)   # dieselbe kanonische Form wie enrich selbst

    # Container: eine Wurzel, unkomprimiert, nur das Inventar + Manifest
    # (Profil handover: keine _history, keine Sperrdateien)
    wurzel = f"{stamm}.enrich"
    puffer = io.BytesIO()
    with zipfile.ZipFile(puffer, "w", zipfile.ZIP_STORED) as z:
        z.write(d.path / "manifest.json", f"{wurzel}/manifest.json")
        for rel in sorted(m.files):
            z.write(d.path / rel, f"{wurzel}/{rel}")
    return puffer.getvalue()


def schicht_origin(schicht: bytes) -> str:
    return json.loads(schicht.decode("utf-8")).get("origin", "mixed")


# ---------- Lesen ----------

def manifest_format2(z: zipfile.ZipFile) -> tuple[str, dict] | None:
    """(Wurzel, Manifest), wenn der Container ein Format-2-Inventar
    trägt — erkannt an `files` + `source`, nicht an `format` (enrich
    schreibt beides schon mit `format: 1`)."""
    for n in z.namelist():
        if n.endswith("/manifest.json") and n.count("/") == 1:
            try:
                m = json.loads(z.read(n).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
            if isinstance(m.get("files"), dict) and m.get("source"):
                return n.split("/", 1)[0], m
            return None
    return None


def inventar_pruefen(z: zipfile.ZipFile, wurzel: str, m: dict) -> None:
    """Jede Datei des Inventars da und unverändert — ausser `_history/`
    (enrich-Entscheid 2026-09-11: Betriebsmittel, reist nicht mit)."""
    verletzt = []
    for pfad, eintrag in m["files"].items():
        if eintrag.get("role") == "history" or pfad.startswith("_history/"):
            continue
        try:
            inhalt = z.read(f"{wurzel}/{pfad}")
        except KeyError:
            verletzt.append(f"{pfad} fehlt"); continue
        if content_hash(inhalt) != eintrag.get("hash"):
            verletzt.append(f"{pfad} verändert")
    if verletzt:
        raise ValueError("Inventar stimmt nicht: " + ", ".join(verletzt))


def lies(z: zipfile.ZipFile, wurzel: str, m: dict) -> dict | None:
    """Format-2-Container → Bibliotheksform. None, wenn die Quelle kein
    Transkript in unserer Form ist (ein enrich-eigenes Transkript-Dossier
    hat `source.canonical = 2-text-raw.json`) — dann greift der
    Format-1-Weg über die Zeitkarte. Die Herkunftsflags werden
    ÜBERNOMMEN, nie eingeebnet (FORMAT.md §5)."""
    inventar_pruefen(z, wurzel, m)
    quelle = m.get("source") or {}
    kanon = quelle.get("canonical") or ""
    try:
        t = json.loads(z.read(f"{wurzel}/{kanon}").decode("utf-8"))
    except (KeyError, ValueError):
        return None
    if not isinstance(t.get("segments"), list):
        return None
    segmente = [{"id": s["id"], "start": s["t0_s"], "end": s["t1_s"],
                 "sprecher": s.get("speaker"), "text": s.get("text", ""),
                 "origin": s.get("origin", "machine")}
                for s in t["segments"]]
    sprecher = [{"id": p["id"], "name": p["name"],
                 "origin": p.get("origin", "machine")}
                for p in t.get("speakers", [])]
    audio_name = audio_bytes = None
    if quelle.get("media"):
        audio_name = Path(quelle["media"]).name
        audio_bytes = z.read(f"{wurzel}/{quelle['media']}")
    journal = [_journal_aus_run(r) for r in m.get("runs", [])
               if r.get("tool") == TOOL]
    return {"name": m.get("title") or t.get("name") or "",
            "segmente": segmente, "sprecher": sprecher,
            "audio_name": audio_name, "audio_bytes": audio_bytes,
            "genau": True, "journal": journal}


def _journal_aus_run(r: dict) -> dict:
    """enrich RunRecord → Bibliotheks-Run (die Kette wird in der
    Bibliothek neu geschlossen; `prev` aus dem Container gilt dort nicht)."""
    who = r.get("who") or {}
    aus = {"id": r["id"], "prev": None, "origin": r.get("origin", "machine"),
           "who": {"app": f"{who.get('app', TOOL)}/{who.get('version', '')}".rstrip("/"),
                   "install": who.get("install"), "user": who.get("user")},
           "started": r.get("started"), "finished": r.get("finished"),
           "layer": "transcript", "did": (r.get("summary") or {}).get("did", ""),
           "result": {"records": len(r.get("changed") or {})}}
    if r.get("changed"):
        aus["changed"] = dict(r["changed"])
    return aus
