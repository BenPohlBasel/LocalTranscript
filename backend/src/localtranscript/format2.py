"""Der .enrich-Container nach FORMAT.md, Format 2.

Das gevendorte enrich-Werkzeug (`enrich_export.textsatz`) setzt aus dem
Transkript ein Format-1-Dossier: PDF, Layout, T0, T1, Zeitkarte — mit
Manifest und Läufen, genau wie enrich selbst. Dieser Modul nimmt das
Ergebnis und macht daraus den Format-2-Container:

- Dateien in die Konvention (Anhang A): source/ text/ — keine Nummern.
- Jede Schicht bekommt den Kopf des Schicht-Vertrags (§2): kind,
  id, origin, from, by, did, result. Der Inhalt bleibt, wie enrich
  ihn schreibt — additiv.
- Das Transkript wird die Quelle: source/transcript.json, Schema
  transcript/1.0.0, mit `origin` je Segment und Sprecher (§5).
- Das Manifest (§3) ist Inventar (`files` mit Hash für JEDE Datei —
  auch das Audio), Lineage (`layers`) und Journal (`runs`, verkettet).
- Profil `handover`: keine _history, Läufe nur, soweit sie einen
  gültigen Stand erzeugt haben — hier: alle, es gibt keine anderen.
- ZIP_STORED, genau ein Wurzelverzeichnis `<name>.enrich/`.

Das Verzeichnis ist enrichs Arbeitsform; das hier ist die Sendung.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

from . import bibliothek
from .config import identitaet, ulid

FORMAT = 2
MANIFEST_SCHEMA = "manifest/1.0.0"
TRANSCRIPT_SCHEMA = "transcript/1.0.0"

#: Format 1 → Konvention Format 2 (FORMAT.md Anhang A)
ALIAS = {
    "0-intake.json": "source/intake.json",
    "1-layout.json": "text/layout.json",
    "2-text-raw.json": "text/raw.json",
    "2z-zeitkarte.json": "text/timemap.json",
    "3-text-clean.json": "text/clean.json",
    "source.pdf": "source/document.pdf",
}
#: kind je Schicht und der eine Satz für Menschen
KIND = {
    "source/intake.json": ("intake", "Quelle geprüft und angenommen."),
    "text/layout.json": ("layout", "Blöcke und Lesefluss des gesetzten PDFs."),
    "text/raw.json": ("text-raw", "T0: Rohtext in Lesereihenfolge, nie normalisiert."),
    "text/clean.json": ("text-clean", "T1: bereinigter Text mit Diff-Map zu T0."),
    "text/timemap.json": ("timemap", "Turns: T1-Offsets ↔ Sekunden ↔ Sprecher, aus dem Transkript abgeleitet."),
}


def _sha(daten: bytes) -> str:
    return "sha256:" + hashlib.sha256(daten).hexdigest()


def _kanon(d: dict) -> bytes:
    return bibliothek.kanonisch(d).encode("utf-8")


def _origin_aus_agent(agent: dict | None) -> str:
    t = (agent or {}).get("type", "derived")
    return {"extracted": "machine", "derived": "machine",
            "llm": "llm", "human": "human"}.get(t, "machine")


def baue_container(daten: dict, dossier_dir: Path, audio: Path | None,
                   stamm: str) -> bytes:
    """Format-1-Dossierverzeichnis (aus textsatz) + kanonisches
    Transkript → Format-2-Container als Bytes."""
    wer = identitaet()
    manifest1 = json.loads((dossier_dir / "manifest.json").read_text("utf-8"))
    runs1 = manifest1.get("runs", [])
    wurzel = f"{stamm}.enrich"

    # ---- Schichten: verschieben, Kopf setzen, IDs vergeben ----
    layer_id: dict[str, str] = {}          # neuer Pfad → lay-ID
    alt_zu_id: dict[str, str] = {}         # alter Name → lay-ID
    for alt, neu in ALIAS.items():
        if (dossier_dir / alt).is_file() and neu.endswith(".json"):
            lid = f"lay-{ulid()}"
            layer_id[neu] = lid
            alt_zu_id[alt] = lid
    transcript_id = f"lay-{ulid()}"

    dateien: dict[str, bytes] = {}          # Pfad im Container → Bytes
    layers: dict[str, dict] = {}
    runs2: list[dict] = []

    def run_fuer(alt: str) -> dict | None:
        for r in reversed(runs1):
            if r.get("layer") == alt:
                return r
        return None

    for alt, neu in ALIAS.items():
        quelle = dossier_dir / alt
        if not quelle.is_file():
            continue
        if not neu.endswith(".json"):
            dateien[neu] = quelle.read_bytes()
            continue
        inhalt = json.loads(quelle.read_text("utf-8"))
        r = run_fuer(alt) or {}
        kind, did = KIND[neu]
        von = {alt_zu_id[a]: h for a, h in (r.get("inputs") or {}).items()
               if a in alt_zu_id}
        if neu == "text/timemap.json":
            von[transcript_id] = "(kanonisch, s. source/transcript.json)"
        kopf = {"kind": kind, "id": layer_id[neu],
                "origin": _origin_aus_agent(inhalt.get("agent")),
                "from": von,
                "by": {"tool": r.get("tool", "enrich-textimport"),
                       "version": r.get("tool_version", ""),
                       "app": wer["app"], "install": wer["install"]},
                "did": did,
                "result": r.get("summary") or {}}
        inhalt = {**kopf, **inhalt}      # Kopf voran, Inhalt bleibt
        dateien[neu] = _kanon(inhalt)
        layers[layer_id[neu]] = {"path": neu, "kind": kind,
                                 "schema": inhalt.get("schema", ""),
                                 "origin": kopf["origin"], "from": von,
                                 "current": True}
        if r:
            runs2.append({"id": r.get("id", f"run-{ulid()}"),
                          "origin": kopf["origin"],
                          "who": {"app": wer["app"], "install": wer["install"],
                                  "tool": r.get("tool"), "version": r.get("tool_version")},
                          "started": r.get("started"), "finished": r.get("finished"),
                          "layer": layer_id[neu], "did": did,
                          "from": von, "result": r.get("summary") or {}})

    # ---- die Quelle: das Transkript ----
    daten = bibliothek._ergaenze_schema1(dict(daten))
    origins = {s.get("origin") for s in daten["segmente"]} | \
              {p.get("origin") for p in daten["sprecher"]}
    transcript = {
        "schema": TRANSCRIPT_SCHEMA, "kind": "transcript", "id": transcript_id,
        "origin": next(iter(origins)) if len(origins) == 1 else "mixed",
        "from": {}, "by": {"app": wer["app"], "install": wer["install"],
                           **({"user": wer["user"]} if wer["user"] else {}),
                           **_whisper_by(daten.get("quelle", {}))},
        "did": _did_transkript(daten),
        "result": {"segments": len(daten["segmente"]),
                   "speakers": len(daten["sprecher"])},
        "name": daten.get("name", stamm),
        "language": daten.get("quelle", {}).get("language"),
        "speakers": [{"id": p["id"], "name": p["name"],
                      "origin": p.get("origin", "machine")}
                     for p in daten["sprecher"]],
        "segments": [{"id": s["id"], "t0_s": s["start"], "t1_s": s["end"],
                      "speaker": s.get("sprecher"), "text": s["text"],
                      "origin": s.get("origin", "machine")}
                     for s in daten["segmente"]],
    }
    dateien["source/transcript.json"] = _kanon(transcript)
    layers[transcript_id] = {"path": "source/transcript.json",
                             "kind": "transcript", "schema": TRANSCRIPT_SCHEMA,
                             "origin": transcript["origin"], "from": {},
                             "current": True}
    # Das Journal der Bibliothek geht ins Manifest — zuerst, denn das
    # Transkript ist die Quelle aller anderen Läufe.
    fuer_manifest = []
    for r in daten.get("journal", []):
        r2 = dict(r)
        r2["layer"] = transcript_id
        fuer_manifest.append(r2)
    runs = fuer_manifest + runs2

    # ---- Audio ----
    media = None
    if audio is not None and audio.is_file():
        media = f"source/audio{audio.suffix.lower()}"
        dateien[media] = audio.read_bytes()

    # ---- Kette: jeder Run trägt den Hash des vorangehenden ----
    vorher = None
    for r in runs:
        r["prev"] = vorher
        vorher = _sha(_kanon(r))

    # ---- Manifest ----
    files = {}
    for pfad, inhalt in dateien.items():
        rolle = ("media" if pfad == media
                 else "rendered" if pfad == "source/document.pdf"
                 else "layer")
        eintrag = {"role": rolle, "hash": _sha(inhalt), "bytes": len(inhalt)}
        if rolle == "layer":
            eintrag["layer"] = layer_id.get(pfad) or transcript_id
        if rolle == "rendered":
            eintrag["from"] = "source/transcript.json"
        files[pfad] = eintrag
    manifest = {
        "format": FORMAT, "schema": MANIFEST_SCHEMA,
        "dossier_id": f"dos-{ulid()}",
        "title": daten.get("name", stamm),
        "created": daten.get("created") or runs[0].get("started") if runs else None,
        "producer": {"app": wer["app"], "install": wer["install"],
                     **({"user": wer["user"]} if wer["user"] else {})},
        "source": {"kind": "transcript", "canonical": "source/transcript.json",
                   **({"media": media} if media else {}),
                   "rendered": "source/document.pdf"},
        "profile": "handover",
        "analyse_kette": manifest1.get("analyse_kette", "narrativ"),
        "license_mode": manifest1.get("license_mode", "full"),
        "gates": manifest1.get("gates", {}),
        "files": files, "layers": layers, "runs": runs,
    }
    manifest_bytes = _kanon(manifest)

    # ---- Container ----
    puffer = io.BytesIO()
    with zipfile.ZipFile(puffer, "w", zipfile.ZIP_STORED) as z:
        z.writestr(f"{wurzel}/manifest.json", manifest_bytes)
        for pfad in sorted(dateien):
            z.writestr(f"{wurzel}/{pfad}", dateien[pfad])
    return puffer.getvalue()


def _whisper_by(quelle: dict) -> dict:
    aus = {}
    if quelle.get("model"):
        aus["model"] = f"whisper {quelle['model']}"
    if quelle.get("diarize"):
        aus["diarization"] = "speechbrain-ecapa"
    return aus


def _did_transkript(daten: dict) -> str:
    art = daten.get("quelle", {}).get("erzeugt", "")
    korrigiert = any(s.get("origin") == "human" for s in daten["segmente"])
    if art == "transcription":
        return ("Whisper-Transkript mit Sprechertrennung"
                + (", danach von Hand korrigiert." if korrigiert else "."))
    return f"Importiert aus {daten.get('quelle', {}).get('datei', '?')}" \
        + (", danach von Hand korrigiert." if korrigiert else ".")


# ---------- Lesen ----------

def ist_format2(z: zipfile.ZipFile) -> dict | None:
    """Manifest, wenn der Container Format 2 ist — sonst None."""
    for n in z.namelist():
        if n.endswith("/manifest.json") and n.count("/") == 1:
            try:
                m = json.loads(z.read(n).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
            return m if m.get("format", 1) >= 2 else None
    return None


def lies(z: zipfile.ZipFile, manifest: dict) -> dict:
    """Format-2-Container → {name, segmente, sprecher, audio_name,
    audio_bytes, genau, journal}. Die Herkunftsflags werden ÜBERNOMMEN,
    nie eingeebnet (FORMAT.md §5)."""
    wurzel = next(n for n in z.namelist() if n.endswith("/manifest.json")).split("/", 1)[0]
    quelle = manifest.get("source", {})
    kanon = quelle.get("canonical", "source/transcript.json")
    try:
        t = json.loads(z.read(f"{wurzel}/{kanon}").decode("utf-8"))
    except KeyError as e:
        raise ValueError(f"Container ohne {kanon}") from e
    # Integrität: jede Datei im Inventar mit passendem Hash
    verletzt = []
    for pfad, eintrag in manifest.get("files", {}).items():
        try:
            inhalt = z.read(f"{wurzel}/{pfad}")
        except KeyError:
            verletzt.append(f"{pfad} fehlt"); continue
        if _sha(inhalt) != eintrag.get("hash"):
            verletzt.append(f"{pfad} verändert")
    if verletzt:
        raise ValueError("Inventar stimmt nicht: " + ", ".join(verletzt))
    segmente = [{"id": s["id"], "start": s["t0_s"], "end": s["t1_s"],
                 "sprecher": s.get("speaker"), "text": s.get("text", ""),
                 "origin": s.get("origin", "machine")}
                for s in t.get("segments", [])]
    sprecher = [{"id": p["id"], "name": p["name"],
                 "origin": p.get("origin", "machine")}
                for p in t.get("speakers", [])]
    audio_name = audio_bytes = None
    media = quelle.get("media")
    if media:
        audio_name = Path(media).name
        audio_bytes = z.read(f"{wurzel}/{media}")
    journal = [r for r in manifest.get("runs", [])
               if r.get("layer") == t.get("id")]
    return {"name": manifest.get("title") or t.get("name") or "",
            "segmente": segmente, "sprecher": sprecher,
            "audio_name": audio_name, "audio_bytes": audio_bytes,
            "genau": True, "journal": journal}
