"""Das .enrich-Paket LESEN — die Gegenrichtung zu exporte._enrich_zip.

Zweck (User 2026-09-09): ein Transkript an Kolleginnen und Kollegen
geben, die im Human-Editor weiterarbeiten. Ein Paket, zwei Fähigkeiten
— enrich füttern UND verlustfrei zurück in die Bibliothek.

REGEL: gezogen werden NUR `transkript.json` und das Audio. Alle
Analyse-Schichten werden verworfen, auch wenn das Dossier sie trägt —
nach dem ersten Edit im Editor stimmte keine davon mehr. Ein Dossier
mit voller enrich-Kette lässt sich also importieren; ankommen tut nur
der Kern.

`transkript.json` ist ein VTT als JSON: dieselben Cues, dazu die
Sprecher als Entitäten mit stabilen ids — das kann VTT nicht.

Fehlt die Beilage (fremdes Dossier oder ein Export von vor dieser
Version), bleibt die Zeitkarte `2z-zeitkarte.json` als Rückfall. Die
kennt nur zusammengefasste Turns, das Ergebnis ist also gröber als das
Original — deshalb sagt der Bericht es LAUT.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

AUDIO_NAMEN = ("audio.mp3", "audio.m4a", "audio.wav", "audio.ogg",
               "audio.flac", "audio.aac", "audio.webm")


class PaketFehler(ValueError):
    """Das Zip ist kein lesbares LocalTranscript-/enrich-Paket."""


def _mitglied(z: zipfile.ZipFile, blatt: str) -> str | None:
    """Mitgliedsname zu einem Dateinamen — der Dossier-Ordner heißt wie
    das Transkript, also nie hart auf einen Pfad setzen."""
    for n in z.namelist():
        if n.rsplit("/", 1)[-1] == blatt:
            return n
    return None


def _aus_zeitkarte(karte: dict,
                   text: str) -> tuple[list[dict], list[dict]]:
    """Rückfall: Zeitkarte + T1 → Segmente + Sprecher.

    Die Zeitkarte trägt nur OFFSETS (`start`/`end`) in den bereinigten
    Text — der Wortlaut kommt aus `3-text-clean.json → streams.main`.
    Ohne ihn kämen leere Segmente an, und das wäre schlimmer als eine
    ehrliche Absage.

    Gröber als das Original: die Einheiten sind Turns, nicht Segmente.
    """
    namen: list[str] = []
    for e in karte.get("einheiten", []):
        wer = (e.get("speaker") or "").strip()
        if wer and wer not in namen:
            namen.append(wer)
    sid = {n: f"sp{i + 1}" for i, n in enumerate(namen)}
    segmente: list[dict] = []
    for e in karte.get("einheiten", []):
        wer = (e.get("speaker") or "").strip()
        a, b = int(e.get("start") or 0), int(e.get("end") or 0)
        wortlaut = " ".join(text[a:b].split()) if text else ""
        # Ältere Dossiers tragen das Label IM Text („Name [00:00:03]:
        # …"). Es ist redundant — der Sprecher steht in derselben
        # Einheit. Abgeräumt wird nur, was zum bekannten Namen passt,
        # nie ein geratenes Präfix.
        if wer:
            wortlaut = re.sub(
                rf"^{re.escape(wer)}\s*(?:\[\d\d:\d\d:\d\d\])?\s*:\s*",
                "", wortlaut)
        if not wortlaut:
            continue
        segmente.append({"start": float(e.get("t0_s") or 0.0),
                         "end": float(e.get("t1_s") or 0.0),
                         "sprecher": sid.get(wer),
                         "text": wortlaut})
    return segmente, [{"id": i, "name": n} for n, i in sid.items()]


def _t1(z: zipfile.ZipFile) -> str:
    """Bereinigter Text (T1) aus dem Dossier — leer, wenn es ihn nicht
    gibt."""
    m = _mitglied(z, "3-text-clean.json")
    if not m:
        return ""
    try:
        d = json.loads(z.read(m).decode("utf-8"))
        return d.get("streams", {}).get("main", "") or ""
    except (ValueError, UnicodeDecodeError, AttributeError):
        return ""


def lies(daten: bytes) -> dict:
    """Paket-Bytes → {name, segmente, sprecher, audio_name, audio_bytes,
    genau} — `genau` sagt, ob die kanonische Beilage gefunden wurde."""
    try:
        z = zipfile.ZipFile(io.BytesIO(daten))
    except zipfile.BadZipFile as e:
        raise PaketFehler(f"Kein lesbares Zip: {e}") from e

    with z:
        tj = _mitglied(z, "transkript.json")
        name = ""
        genau = tj is not None
        if tj:
            try:
                kanon = json.loads(z.read(tj).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                raise PaketFehler(
                    f"transkript.json ist beschädigt: {e}") from e
            segmente = kanon.get("segmente") or []
            sprecher = kanon.get("sprecher") or []
            name = kanon.get("name") or ""
            if not segmente:
                raise PaketFehler("transkript.json ohne Segmente")
        else:
            zk = _mitglied(z, "2z-zeitkarte.json")
            if not zk:
                raise PaketFehler(
                    "Weder transkript.json noch 2z-zeitkarte.json im "
                    "Paket — das ist kein LocalTranscript- oder "
                    "enrich-Transkript.")
            segmente, sprecher = _aus_zeitkarte(
                json.loads(z.read(zk).decode("utf-8")), _t1(z))
            if not segmente:
                raise PaketFehler(
                    "Zeitkarte ohne lesbare Einheiten — ohne "
                    "3-text-clean.json bleibt kein Wortlaut übrig.")

        if not name:
            # Dossier-Ordner heißt wie das Transkript: "<name>.enrich/"
            wurzel = z.namelist()[0].split("/", 1)[0]
            name = wurzel.removesuffix(".enrich") or "Transkript"

        audio_name = audio_bytes = None
        for blatt in AUDIO_NAMEN:
            m = _mitglied(z, blatt)
            if m:
                audio_name, audio_bytes = blatt, z.read(m)
                break

    return {"name": name, "segmente": segmente, "sprecher": sprecher,
            "audio_name": audio_name, "audio_bytes": audio_bytes,
            "genau": genau}


def ist_paket(pfad_oder_name: str) -> bool:
    """Endung, die `lies` versuchen darf."""
    return Path(pfad_oder_name).suffix.lower() == ".zip"
