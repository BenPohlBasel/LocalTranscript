"""Abgeleitete Exporte aus dem kanonischen Modell.

vtt/csv/txt: reine Text-Renderer (ausgabe.py).
enrich: vollwertiges enrich-Dossier-Zip — Segmente → Turns →
turns_zu_struktur → baue_struktur_dossier (gevendorte enrich-Bausteine
+ enrich-core): T0=T1 byte-treu, PDF in Recursive gesetzt,
Zeitkarte 2z (T1-Offsets ↔ Sekunden ↔ Sprecher), Audio-Kopie,
analyse_kette=narrativ. Genau das Dossier, das enrichs
Text/Transkript-Import aus vtt+Audio bauen würde (User-Auftrag) —
enrichs Dossier-Import nimmt das Zip direkt (Dossier.unpack).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from . import ausgabe, bibliothek

FORMATE = ("vtt", "csv", "txt", "enrich")


def export_bytes(eid: str, format: str) -> tuple[bytes, str, str]:
    """(Inhalt, Dateiname, media_type) — für Browser-Download und
    Datei-Schreiben gleichermaßen."""
    daten = bibliothek.lese(eid)
    seg = bibliothek.export_segmente(daten)
    stamm = bibliothek._slug(daten["name"])
    if format == "vtt":
        return (ausgabe.build_vtt(seg).encode("utf-8"),
                f"{stamm}.vtt", "text/vtt")
    if format == "csv":
        return (ausgabe.build_csv(seg).encode("utf-8-sig"),
                f"{stamm}.csv", "text/csv")
    if format == "txt":
        return (ausgabe.build_txt(seg).encode("utf-8"),
                f"{stamm}.txt", "text/plain")
    if format == "enrich":
        return (_enrich_zip(eid, daten, seg, stamm),
                f"{stamm}.enrich.zip", "application/zip")
    raise ValueError(f"Unbekanntes Format: {format}")


def _enrich_zip(eid: str, daten: dict, seg: list[dict],
                stamm: str) -> bytes:
    from .enrich_export.textsatz import baue_struktur_dossier
    from .enrich_export.turns import turns_zu_struktur

    turns = [{"t0_s": s["start"], "t1_s": s["end"],
              "speaker": s["sprecher"], "text": s["text"]}
             for s in seg if s["text"].strip()]
    if not turns:
        raise ValueError("Leeres Transkript — nichts zu exportieren")
    struktur, zeiten = turns_zu_struktur(turns)
    audio = bibliothek.audio_pfad(eid)
    with tempfile.TemporaryDirectory() as td:
        dp = Path(td) / f"{stamm}.enrich"
        d, _bericht = baue_struktur_dossier(
            dp, struktur, quelle=daten["name"], user="localtranscript",
            zeiten=zeiten, audio=audio,
            zeiten_quelle="localtranscript")
        d.set_analyse_kette("narrativ", "localtranscript")
        zip_pfad = Path(td) / f"{stamm}.enrich.zip"
        d.pack(zip_pfad)
        return zip_pfad.read_bytes()
