# VENDORED-Extrakt aus enrich@d1f4214 —
# packages/enrich-serve/src/enrich_serve/textimport.py (die puren
# Transkript-Funktionen, VERBATIM-Slice; Drift-Guard:
# tests/test_drift_guard.py. NIE formatieren/fixen (ruff-exclude)).
from __future__ import annotations

import csv
import io
import re

def _zeit_s(roh: str) -> float:
    """"MM:SS" | "HH:MM:SS" | "HH:MM:SS.mmm" → Sekunden."""
    teile = roh.strip().split(":")
    try:
        s = float(teile[-1])
        m = int(teile[-2]) if len(teile) > 1 else 0
        h = int(teile[-3]) if len(teile) > 2 else 0
        return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return 0.0


def parse_transkript_csv(daten: bytes) -> list[dict]:
    """LocalTranscript-CSV → Turns [{t0_s, t1_s, speaker, text}]."""
    text = daten.decode("utf-8-sig", errors="replace")
    zeilen = list(csv.reader(io.StringIO(text)))
    if not zeilen:
        return []
    kopf = [z.strip().casefold() for z in zeilen[0]]

    def spalte(*namen: str) -> int | None:
        for i, k in enumerate(kopf):
            if any(n in k for n in namen):
                return i
        return None

    c_in = spalte("time-in", "start", "begin")
    c_out = spalte("time-out", "end")
    c_sp = spalte("speaker", "sprecher")
    c_tx = spalte("text", "inhalt")
    if c_tx is None:
        raise ValueError("CSV ohne Text-Spalte")
    turns = []
    for zeile in zeilen[1:]:
        if len(zeile) <= c_tx or not zeile[c_tx].strip():
            continue
        turns.append({
            "t0_s": _zeit_s(zeile[c_in]) if c_in is not None
            and len(zeile) > c_in else 0.0,
            "t1_s": _zeit_s(zeile[c_out]) if c_out is not None
            and len(zeile) > c_out else 0.0,
            "speaker": (zeile[c_sp].strip() if c_sp is not None
                        and len(zeile) > c_sp else ""),
            "text": " ".join(zeile[c_tx].split())})
    for t in turns:
        if t["t1_s"] < t["t0_s"]:
            t["t1_s"] = t["t0_s"]
    return turns


_VTT_ZEIT = re.compile(
    r"(\d{1,2}:)?\d{1,2}:\d{1,2}[.,]\d{1,3}\s*-->\s*"
    r"((\d{1,2}:)?\d{1,2}:\d{1,2}[.,]\d{1,3})")
_VTT_SPRECHER = re.compile(r"^(?:<v\s+([^>]+)>|([^:<>\n]{1,40}):\s+)")


def parse_transkript_vtt(daten: bytes) -> list[dict]:
    """WEBVTT → Turns; Sprecher-Präfix ("Speaker n: " oder <v …>)
    startet einen Turn, präfixlose Cues setzen ihn fort."""
    text = daten.decode("utf-8-sig", errors="replace")
    turns: list[dict] = []
    cue_zeit: tuple[float, float] | None = None
    for zeile in text.splitlines():
        z = zeile.strip()
        m = _VTT_ZEIT.match(z)
        if m:
            a, b = z.split("-->")
            cue_zeit = (_zeit_s(a.replace(",", ".")),
                        _zeit_s(b.split()[0].replace(",", ".")))
            continue
        if not z or z == "WEBVTT" or z.isdigit() \
                or z.startswith(("NOTE", "STYLE", "REGION")):
            continue
        if cue_zeit is None:
            continue
        sp = _VTT_SPRECHER.match(z)
        inhalt = z[sp.end():].strip() if sp else z
        sprecher = (sp.group(1) or sp.group(2) or "").strip() \
            if sp else ""
        if sp:
            # Sprecher-Präfix = IMMER neuer Turn (LocalTranscript
            # setzt das Präfix nur am Turn-Anfang; auch derselbe
            # Sprecher beginnt damit einen neuen Beitrag)
            turns.append({"t0_s": cue_zeit[0], "t1_s": cue_zeit[1],
                          "speaker": sprecher, "text": inhalt})
        elif turns:
            turns[-1]["text"] = (turns[-1]["text"] + " "
                                 + inhalt).strip()
            turns[-1]["t1_s"] = max(turns[-1]["t1_s"], cue_zeit[1])
        else:
            turns.append({"t0_s": cue_zeit[0], "t1_s": cue_zeit[1],
                          "speaker": "", "text": inhalt})
    return [t for t in turns if t["text"]]


def _tc(sekunden: float) -> str:
    """IMMER hh:mm:ss (User 2026-08-30: „unbedingt") — eindeutig,
    kein Formatwechsel mitten im Dokument."""
    s = int(sekunden)
    h, rest = divmod(s, 3600)
    m, sek = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{sek:02d}"


def turns_zu_struktur(turns: list[dict]) -> tuple[dict, list[dict]]:
    """Turns -> (Struktur fuers Setzen, Zeit-Spannen in MAIN-Offsets).

    UMBAU 2026-08-30 (User: "Sprecher und Timecode gehoeren nicht in
    den Textstrom"): aufeinanderfolgende Turns desselben Sprechers
    werden zu EINEM Block verschmolzen (wie im CSV-Export), und
    Sprecher+Timecode werden als eigene LABEL-ZEILE ueber dem Absatz
    gesetzt (typ "sprecher" -> Block speaker-label -> Strom other) —
    sichtbar im PDF, aber main traegt REINE REDE. Die Zeitkarte
    referenziert die Rede-Offsets; Attribution fuer die Analyse kommt
    strukturiert aus ihr (kette.mit_sprecher_marken), nie aus dem
    Text."""
    def run(text: str, *, fett: bool = False,
            farbe: str = "") -> dict:
        return {"text": text, "fett": fett, "kursiv": False,
                "farbe": farbe, "groesse": 0.0, "mono": False}

    # 1. Verschmelzen: gleicher Sprecher hintereinander = EIN Block
    bloecke: list[dict] = []
    for t in turns:
        text = " ".join(t["text"].split())
        if not text:
            continue
        if bloecke and bloecke[-1]["speaker"] == t["speaker"]:
            bloecke[-1]["text"] += " " + text
            bloecke[-1]["t1_s"] = max(bloecke[-1]["t1_s"], t["t1_s"])
        else:
            bloecke.append({"t0_s": t["t0_s"], "t1_s": t["t1_s"],
                            "speaker": t["speaker"], "text": text})

    absaetze = []
    zeiten = []
    pos = 0  # NUR main-Offsets — Label-Zeilen leben im other-Strom
    for i, b in enumerate(bloecke):
        tc = f"[{_tc(b['t0_s'])}]" if (b["t0_s"] or b["t1_s"]) else ""
        label_runs = []
        if b["speaker"]:
            label_runs.append(run(b["speaker"], fett=True))
        if tc:
            label_runs.append(run(f" {tc}" if b["speaker"] else tc,
                                  farbe="#8a8a8a"))
        if label_runs:
            absaetze.append({"typ": "sprecher", "runs": label_runs,
                             "noten": []})
        if i:
            pos += 1  # Absatz-Trenner in main
        start = pos
        absaetze.append({"typ": "paragraph", "runs": [run(b["text"])],
                         "noten": []})
        pos += len(b["text"])
        zeiten.append({"start": start, "end": pos,
                       "t0_s": b["t0_s"], "t1_s": b["t1_s"],
                       "speaker": b["speaker"]})
    return ({"absaetze": absaetze, "fussnoten": [], "endnoten": []},
            zeiten)


def text_zu_struktur(text: str) -> dict:
    """Plain Text: jede Zeile ein Absatz (Transkript-Konvention)."""
    absaetze = []
    for zeile in text.split("\n"):
        if not zeile.strip():
            continue
        absaetze.append({"typ": "paragraph", "runs": [
            {"text": " ".join(zeile.split()), "fett": False,
             "kursiv": False, "farbe": "", "groesse": 0.0,
             "mono": False}], "noten": []})
    return {"absaetze": absaetze, "fussnoten": [], "endnoten": []}


