"""Der .enrich-Container nach FORMAT.md, Format 2 — enrich-core schreibt,
LocalTranscript ergänzt.

Das gevendorte enrich-Werkzeug (`enrich_export.textsatz`) setzt aus dem
Transkript das Dossier: Transkript-Schicht (die QUELLE, `source/
transcript.json`), PDF (`rendered`), Layout, T0, T1, Zeitkarte — und
enrich-core führt dabei das Manifest nach Format 2 (Inventar `files`,
Lineage `layers`, `source`, Journal mit `prev`-Kette, Namenskonvention
Anhang A). Dieser Modul legt darauf NUR, was enrich nicht wissen kann
(Stand enrich@5e169b8, 2026-09-11):

- Die Transkript-Schicht selbst — aus der Bibliotheksform, mit `origin`
  je Segment und Sprecher und dem `by` (Whisper-Modell, Sprechertrennung,
  Person). Den Kopf (§2) füllt enrichs `write_layer`.
- Das Journal der Bibliothek — Whisper-Lauf, Import, Editor-Sitzungen —
  als schlanke `RunRecord`s (wann, was, wer, wo; FORMAT.md §3.1) VOR
  den Läufen des Textsatzes (das Transkript ist die Quelle aller anderen
  Schichten); die Kette wird über alle Einträge neu geschlossen.
- `producer` = LocalTranscript, `title`. Die Sendung selbst baut enrichs
  `Dossier.pack(profile="handover")`: eine Wurzel, unkomprimiert, ohne
  `_history/`, Inventar vollständig, Kette in sich geschlossen.

Der Kompatibilitätstest öffnet den Container mit `Dossier.uebernehmen`
und prüft Inventar und Kette (tests/test_format2.py).
"""
from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

from enrich_core.canonical import content_hash
from enrich_core.dossier import TRANSCRIPT_LAYER, Dossier, run_eintrag_hash
from enrich_core.herkunft import LayerBy
from enrich_core.ids import new_id
from enrich_core.pfade import ALT_ALIASSE
from enrich_core.schemas.manifest import Producer, RunRecord, Who
from enrich_core.schemas.transcript import Segment, Speaker, TranscriptLayer

from . import bibliothek
from .config import APP_VERSION, identitaet

TOOL = "localtranscript"


# ---------- Schreiben ----------

def transkript_schicht(daten: dict, wer: dict) -> TranscriptLayer:
    """Bibliotheksform → Transkript-Schicht (FORMAT.md §5): Sprecher und
    Segmente MIT ihrem `origin`; `by` sagt, womit das Transkript entstand.
    `origin` der Schicht leitet enrich aus den Records ab (`mixed`)."""
    daten = bibliothek._ergaenze_schema1(dict(daten))
    quelle = daten.get("quelle", {})
    # Womit? Der erste Run des Journals weiss es (Whisper, Modell,
    # Sprechertrennung) — sonst nur das, was die Quelle sagt
    erster = next(iter(daten.get("journal") or []), {})
    by = dict(erster.get("by") or {})
    return TranscriptLayer(
        name=daten.get("name", ""), language=quelle.get("language"),
        by=LayerBy(tool=TOOL, version=APP_VERSION,
                   model=(f"{by['tool']} {by['model']}" if by.get("model")
                          else (f"whisper {quelle['model']}" if quelle.get("model")
                                else None)),
                   user=wer.get("user"), diarization=by.get("diarization"),
                   config={k: quelle[k] for k in ("model", "language", "diarize")
                           if k in quelle}),
        did=_did(daten),
        speakers=[Speaker(id=p["id"], name=p["name"],
                          origin=p.get("origin", "machine"))
                  for p in daten["sprecher"]],
        segments=[Segment(id=s["id"], t0_s=s["start"], t1_s=s["end"],
                          speaker=s.get("sprecher"), text=s["text"],
                          origin=s.get("origin", "machine"))
                  for s in daten["segmente"]])


def _did(daten: dict) -> str:
    art = daten.get("quelle", {}).get("erzeugt", "")
    korrigiert = any(s.get("origin") == "human" for s in daten["segmente"])
    kern = ("Whisper-Transkript mit Sprechertrennung" if art == "transcription"
            else f"Importiert aus {daten.get('quelle', {}).get('datei', '?')}")
    return kern + (", danach von Hand korrigiert." if korrigiert else ".")


def _run_aus_journal(r: dict, wer: dict, layer_id: str) -> RunRecord:
    """Bibliotheks-Run → schlanker enrich-Run: wann, was, wer, wo
    (FORMAT.md §3.1). Woraus/womit/was herauskam steht im Schicht-Kopf."""
    who = dict(r.get("who") or {})
    by = dict(r.get("by") or {})
    # Bibliothek führt `app` als «localtranscript/2.2.0»; enrichs Who
    # trennt app und version
    version = (who.get("app") or f"{TOOL}/{APP_VERSION}").split("/", 1)[-1]
    return RunRecord(
        id=r.get("id") or new_id("run"), layer=layer_id, path=TRANSCRIPT_LAYER,
        started=r.get("started") or bibliothek._jetzt(),
        finished=r.get("finished"), did=r.get("did") or None,
        origin=r.get("origin", "machine"),
        who=Who(app=TOOL, version=version,
                install=who.get("install", wer.get("install")),
                user=who.get("user"), tool=by.get("tool"),
                model=by.get("model")),
        changed=dict(r.get("changed") or {}),
    )


def baue_container(daten: dict, d: Dossier, stamm: str) -> bytes:
    """Dossier (von textsatz gebaut, Transkript als Quelle) → Sendung."""
    wer = identitaet()
    m = d.manifest
    ptr = m.current.get(TRANSCRIPT_LAYER)
    layer_id = ptr.layer if ptr is not None and ptr.layer else TRANSCRIPT_LAYER
    # Journal: Bibliothek zuerst, dann der Textsatz; Kette neu schliessen
    eigene = [_run_aus_journal(r, wer, layer_id)
              for r in daten.get("journal", [])]
    alle = eigene + list(m.runs)
    vorher: RunRecord | None = None
    for run in alle:
        run.prev = run_eintrag_hash(vorher) if vorher is not None else None
        vorher = run
    m.runs = alle
    m.producer = Producer(app=TOOL, version=APP_VERSION)
    m.title = daten.get("name") or stamm
    d._write_manifest(m)   # dieselbe kanonische Form wie enrich selbst
    # Die Sendung baut enrich selbst (FORMAT.md §1/§4): eine Wurzel
    # `<stamm>.enrich/`, ZIP_STORED, Profil handover — Quelle + Text-
    # Schichten, ohne _history, Inventar vervollständigt (Audio), Kette
    # über die mitreisenden Runs geschlossen.
    with tempfile.TemporaryDirectory() as td:
        ziel = d.pack(Path(td) / f"{stamm}.enrich", profile="handover")
        return _producer_setzen(ziel.read_bytes(), f"{stamm}.enrich")


def _producer_setzen(sendung: bytes, wurzel: str) -> bytes:
    """`producer` in der Sendung auf LocalTranscript stellen.

    enrichs `_kopf_manifest` überschreibt jeden fremden `producer` mit
    «enrich» — auch in der Kopie, die `handover_plan` baut (Befund
    2026-09-11, enrich BACKLOG 000). FORMAT.md §3 meint aber die
    Anwendung, die den Container ERZEUGT hat; enrich-core ist hier die
    Bibliothek. Das Manifest steht nicht im Inventar, die Kette hängt an
    den Run-Einträgen — die Korrektur berührt keinen Hash."""
    from enrich_core.canonical import canonical_dumps
    alt = zipfile.ZipFile(io.BytesIO(sendung))
    aus = io.BytesIO()
    with zipfile.ZipFile(aus, "w", zipfile.ZIP_STORED) as neu:
        for i in alt.infolist():
            roh = alt.read(i.filename)
            if i.filename == f"{wurzel}/manifest.json":
                m = json.loads(roh.decode("utf-8"))
                m["producer"] = {"app": TOOL, "version": APP_VERSION}
                roh = canonical_dumps(m).encode("utf-8")
            neu.writestr(i, roh)
    return aus.getvalue()


# ---------- Lesen ----------

def manifest_format2(z: zipfile.ZipFile) -> tuple[str, dict] | None:
    """(Wurzel, Manifest), wenn der Container ein Format-2-Inventar
    trägt — erkannt an `files` + `source`, nicht an `format` (enrich
    schrieb beides eine Weile mit `format: 1`)."""
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


def _ist_unser_run(r: dict) -> bool:
    """Schlanke Runs sagen es in `who.app`, ältere in `tool`."""
    return ((r.get("who") or {}).get("app") == TOOL
            or r.get("tool") == TOOL)


def lies(z: zipfile.ZipFile, wurzel: str, m: dict) -> dict | None:
    """Format-2-Container → Bibliotheksform. None, wenn die Quelle kein
    Transkript ist (ein enrich-Dokument-Dossier hat `source.canonical =
    text/raw.json`) — dann greift der Format-1-Weg über die Zeitkarte.
    Die Herkunftsflags werden ÜBERNOMMEN, nie eingeebnet (FORMAT.md §5).
    Gelesen wird der Name, den das Manifest nennt — die Konvention
    (`source/transcript.json`) wie der ältere (`transcript.json`)."""
    inventar_pruefen(z, wurzel, m)
    quelle = m.get("source") or {}
    kanon = quelle.get("canonical") or ""
    if kanon not in (TRANSCRIPT_LAYER, *ALT_ALIASSE):
        return None
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
               if _ist_unser_run(r)]
    zotero = None
    if "source/zotero.json" in m.get("files", {}):
        # Die Zotero-Schicht kommt mit — sie ist Quelle (FORMAT.md §5),
        # ein Mensch hat sie im Ursprung gewählt, hier wird nichts geraten
        zl = json.loads(z.read(f"{wurzel}/source/zotero.json").decode("utf-8"))
        zotero = {k: zl.get(k) for k in (
            "item_key", "citekey", "item_type", "title", "date", "year",
            "publication", "doi", "abstract", "select_link", "creators")}
    return {"name": m.get("title") or t.get("name") or "",
            "segmente": segmente, "sprecher": sprecher,
            "audio_name": audio_name, "audio_bytes": audio_bytes,
            "genau": True, "journal": journal, "zotero": zotero}


def _journal_aus_run(r: dict) -> dict:
    """enrich RunRecord (schlank oder alt) → Bibliotheks-Run (die Kette
    wird in der Bibliothek neu geschlossen; `prev` aus dem Container gilt
    dort nicht)."""
    who = r.get("who") or {}
    aus = {"id": r["id"], "prev": None, "origin": r.get("origin", "machine"),
           "who": {"app": f"{who.get('app', TOOL)}/{who.get('version', '')}".rstrip("/"),
                   "install": who.get("install"), "user": who.get("user")},
           "started": r.get("started"), "finished": r.get("finished"),
           "layer": "transcript",
           "did": r.get("did") or (r.get("summary") or {}).get("did", ""),
           "result": {"records": len(r.get("changed") or {})}}
    if r.get("changed"):
        aus["changed"] = dict(r["changed"])
    if who.get("tool"):
        aus["by"] = {"tool": who["tool"], "model": who.get("model")}
    return aus
