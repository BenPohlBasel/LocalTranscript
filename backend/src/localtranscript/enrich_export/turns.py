# VENDORED-Extrakt aus enrich@3d2b131 —
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
    """Turns -> (Struktur fuers Setzen, Zeit-Spannen in T0-Offsets).

    Absatz = "Sprecher [00:02]: Text" (User 2026-08-30: Timecode
    hinter den Sprecher) — Sprecher fett, Timecode dezent grau; die
    Offsets werden deterministisch mitgerechnet (Absatz-Trenner ist
    das Newline, das der Setzer schreibt)."""
    absaetze = []
    zeiten = []
    pos = 0

    def run(text: str, *, fett: bool = False,
            farbe: str = "") -> dict:
        return {"text": text, "fett": fett, "kursiv": False,
                "farbe": farbe, "groesse": 0.0, "mono": False}

    for i, t in enumerate(turns):
        if i:
            pos += 1  # Absatz-Trenner
        runs = []
        start = pos
        tc = f"[{_tc(t['t0_s'])}]" if (t["t0_s"] or t["t1_s"]) else ""
        if t["speaker"] and tc:
            runs += [run(f"{t['speaker']} ", fett=True),
                     run(tc, farbe="#8a8a8a"),
                     run(": ", fett=True)]
            pos += len(t["speaker"]) + 1 + len(tc) + 2
        elif t["speaker"]:
            runs.append(run(f"{t['speaker']}: ", fett=True))
            pos += len(t["speaker"]) + 2
        elif tc:
            runs.append(run(f"{tc} ", farbe="#8a8a8a"))
            pos += len(tc) + 1
        runs.append(run(t["text"]))
        pos += len(t["text"])
        absaetze.append({"typ": "paragraph", "runs": runs,
                         "noten": []})
        zeiten.append({"start": start, "end": pos,
                       "t0_s": t["t0_s"], "t1_s": t["t1_s"],
                       "speaker": t["speaker"]})
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


# ---------- Import ----------
