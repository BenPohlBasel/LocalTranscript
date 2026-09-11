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

FORMATE = ("vtt", "csv", "txt", "enrich", "qdpx")


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
        # EINE Datei mit Endung .enrich — ein Zip ohne Kompression, wie
        # .docx oder .qdpx (User 2026-09-10). enrich öffnet sie am Inhalt
        # (is_zipfile), macOS zeigt sie als Datei, egal ob enrich.app
        # installiert ist. Das Verzeichnis-Dossier bleibt enrichs
        # Arbeitsform; das hier ist die Weitergabeform.
        return (_enrich_paket(eid, daten, seg, stamm),
                f"{stamm}.enrich", "application/zip")
    if format == "qdpx":
        from . import qdpx
        if not seg:
            raise ValueError("Leeres Transkript — nichts zu exportieren")
        # Bleibt .qdpx.zip: das Archiv enthält <Name>.qdpx UND daneben
        # <Name> Media/ mit dem Audio — so exportiert ATLAS.ti selbst,
        # das Audio liegt per Standard AUSSERHALB des .qdpx (relative:///).
        # Ein Zip im Zip, darum heisst das äussere ehrlich .zip.
        return (qdpx.baue_zip(stamm, seg, daten.get("sprecher", []),
                              bibliothek.audio_pfad(eid)),
                f"{stamm}.qdpx.zip", "application/zip")
    raise ValueError(f"Unbekanntes Format: {format}")


def _enrich_paket(eid: str, daten: dict, seg: list[dict],
                  stamm: str) -> bytes:
    from .enrich_export.textsatz import baue_struktur_dossier
    from .enrich_export.turns import turns_zu_struktur

    # User 2026-08-30: ins Dossier gehen die ZUSAMMENGEFASSTEN
    # Sprecher-Blöcke (wie im CSV), nie einzelne VTT-Zeilen — ein Turn
    # = ein Absatz reiner Rede mit EINER Label-Zeile darüber
    turns = [{"t0_s": t["start"], "t1_s": t["end"],
              "speaker": t["sprecher"], "text": t["text"]}
             for t in ausgabe._turns(seg)]
    if not turns:
        raise ValueError("Leeres Transkript — nichts zu exportieren")
    struktur, zeiten = turns_zu_struktur(turns)
    audio = bibliothek.audio_pfad(eid)
    with tempfile.TemporaryDirectory() as td:
        # User-Regel 2026-08-30: im .enrich-Dossier liegt IMMER mp3
        # (nie wav — enrich-Dossiers sollen nicht aufgebläht sein);
        # schlägt ffmpeg fehl, geht das Original ehrlich mit.
        if audio is not None and audio.suffix.lower() != ".mp3":
            import subprocess

            from .config import get_ffmpeg_cli
            mp3 = Path(td) / "audio.mp3"
            r = subprocess.run(
                [get_ffmpeg_cli(), "-y", "-i", str(audio),
                 "-c:a", "libmp3lame", "-q:a", "2", str(mp3)],
                capture_output=True)
            if r.returncode == 0 and mp3.is_file():
                audio = mp3
        dp = Path(td) / f"{stamm}.enrich"
        # Wer steht im Journal des Dossiers: die E-Mail aus den
        # Einstellungen, wenn eine hinterlegt ist — sonst die App. Die
        # Einstellungen sagen, dass die Adresse hier landet.
        from .config import identitaet
        wer = identitaet()
        # Kopfzeile auf Seite 1 des gesetzten PDFs (enrich@24333d4):
        # «Titel · Datum · Interviewer:in · Citekey». Heute liefert das
        # Transkript Titel und Datum; Interviewer:in und Citekey kommen,
        # sobald die Zotero-Schicht in LocalTranscript entsteht.
        from .enrich_export.textsatz import kopfzeile_aus_meta
        zot = daten.get("zotero")
        meta = ({"title": zot.get("title") or daten.get("name") or stamm,
                 "date": zot.get("date") or zot.get("year") or (daten.get("created") or "")[:10],
                 "creators": zot.get("creators") or [],
                 "citekey": zot.get("citekey")}
                if zot else
                {"title": daten.get("name") or stamm,
                 "date": (daten.get("created") or "")[:10]})
        # Die Quelle des Dossiers ist das Transkript (FORMAT.md §5): der
        # Setzer schreibt es als ERSTE Schicht, leitet die Zeitkarte daraus
        # ab und trägt das PDF als `rendered` ein.
        from .format2 import baue_container, transkript_schicht
        d, _bericht = baue_struktur_dossier(
            dp, struktur, quelle=daten["name"],
            user=wer["user"] or wer["app"],
            zeiten=zeiten, audio=audio,
            zeiten_quelle="localtranscript",
            kopfzeile=kopfzeile_aus_meta(meta),
            transkript=transkript_schicht(daten, wer))
        d.set_analyse_kette("narrativ", "localtranscript")
        if zot:
            # Die Zotero-Schicht über enrichs EINEN Schreibweg —
            # registriert Schicht, Run und Inventar (source/zotero.json)
            from enrich_core.zotero import write_zotero_layer

            from .config import APP_VERSION
            write_zotero_layer(d, dict(zot, collections=[]), force=True,
                               tool="localtranscript", tool_version=APP_VERSION)
        # Journal der Bibliothek davor, producer/title, dann die Sendung
        # (Profil handover) über enrichs eigenen Packer.
        return baue_container(daten, d, stamm)


def packe_verzeichnis(ordner: Path) -> bytes:
    """Verzeichnis → Zip-Bytes, unkomprimiert, mit dem Ordnernamen als
    einziger Wurzel. Dient dem Export und dem Import eines Dossiers,
    das als Verzeichnis (macOS-Package) vorliegt."""
    import io
    import zipfile
    puffer = io.BytesIO()
    with zipfile.ZipFile(puffer, "w", zipfile.ZIP_STORED) as zf:
        for fp in sorted(ordner.rglob("*")):
            if fp.is_file():
                zf.write(fp, f"{ordner.name}/{fp.relative_to(ordner)}")
    return puffer.getvalue()
