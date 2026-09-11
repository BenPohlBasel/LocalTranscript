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
    """Das Dossier: Transkript (Quelle, FORMAT.md §5) + Audio + Zotero.

    Seit enrich f871d33 darf das gesetzte PDF fehlen — «Transkript +
    Audio sind ein gültiger Eingangszustand, die empfangende Anwendung
    setzt die Lesefassung selbst» (enrich tut es beim Import). Damit
    entfällt hier der Setzer samt PyMuPDF und Fonts (Entscheid
    2026-09-11): der Container ist klein, und LocalTranscript hängt an
    enrich-core, nicht mehr an enrich-serve."""
    import shutil

    from enrich_core.dossier import TRANSCRIPT_LAYER, Dossier, utc_now
    from enrich_core.ids import new_id
    from enrich_core.pfade import schicht_datei
    from enrich_core.schemas.common import Agent
    from enrich_core.schemas.manifest import RunRecord, SourceInfo, Who

    from .config import APP_VERSION, identitaet
    from .format2 import TOOL, baue_container, transkript_schicht

    if not seg:
        raise ValueError("Leeres Transkript — nichts zu exportieren")
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
        wer = identitaet()
        # Die Quelle steht beim Anlegen fest (FORMAT.md §3): das
        # Transkript; das Audio gehört zur Quelle (`source/audio.mp3`,
        # Anhang A) und steht als `media` im Manifest.
        media = None
        if audio is not None and audio.is_file():
            media = f"audio{audio.suffix.lower()}"
        d = Dossier.create(dp, quelle=SourceInfo(
            kind="transcript", canonical=TRANSCRIPT_LAYER, media=media))
        if media:
            ziel = schicht_datei(d.path, media)
            ziel.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audio, ziel)
        # Das Transkript wird nicht gerechnet, sondern ÜBERNOMMEN —
        # darum `extracted`; jedes Segment trägt sein eigenes `origin`
        # (FORMAT.md §5), der Kopf wird `mixed`, sobald ein Mensch
        # etwas angefasst hat. `by` (Whisper-Modell, Sprechertrennung,
        # Person) bringt die Schicht mit — enrich respektiert es.
        d.write_layer(TRANSCRIPT_LAYER, transkript_schicht(daten, wer),
                      RunRecord(id=new_id("run"), tool=TOOL,
                                tool_version=APP_VERSION,
                                layer=TRANSCRIPT_LAYER,
                                agent=Agent(type="extracted",
                                            tool=f"{TOOL}/{APP_VERSION}"),
                                who=Who(app=TOOL, version=APP_VERSION,
                                        install=wer["install"],
                                        user=wer["user"]),
                                started=utc_now()))
        d.inventar_pflegen()           # das Audio ins Inventar (Hash)
        d.set_analyse_kette("narrativ", "localtranscript")
        zot = daten.get("zotero")
        if zot:
            # Die Zotero-Schicht über enrichs EINEN Schreibweg —
            # registriert Schicht, Run und Inventar (source/zotero.json);
            # die Kopfzeile «Titel · Datum · Interviewer:in · Citekey»
            # setzt enrich daraus, wenn es die Lesefassung baut.
            from enrich_core.zotero import write_zotero_layer
            write_zotero_layer(d, dict(zot, collections=[]), force=True,
                               tool=TOOL, tool_version=APP_VERSION)
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
