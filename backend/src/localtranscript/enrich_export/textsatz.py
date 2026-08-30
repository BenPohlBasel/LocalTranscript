# VENDORED aus enrich (privates Repo BenPohlBasel/PDFenrichCLI),
# Stand enrich@c2f4358 — packages/enrich-serve/src/enrich_serve/textsatz.py.
# Einzige Abweichung: der Import von _FONT_DIR/_sichtbar zeigt auf
# .schrift (lokaler Extrakt aus refi_text.py). Drift-Guard:
# tests/test_drift_guard.py. NIE formatieren/fixen (ruff-exclude).
"""Stil-Renderer + Dossier-Bau für Struktur-Dokumente (docx/txt/
Transkripte) — Plan docx-import.md, User-Go 2026-08-30.

Leitregel wie refi_text: „Text ist Wahrheit, PDF ist gerenderte Sicht
mit mitgeborener Geometrie-Karte" — T0 ist UNSERE deterministische
Serialisierung der Struktur (es gibt keinen fremden kanonischen Plain
Text), das PDF wird in Recursive gesetzt (Stil-TREUE: fett/kursiv/
Farben/Größen/Mono — nie Font-Identität). Ströme wie die PDF-Pipeline:
main (+ note_markers, Marker-Ziffern hochgestellt; T1 entfernt sie
über note-marker-DiffOps), footnotes (Fuß- + Endnoten als NOTENSEITEN
am Ende — User-Entscheid), captions, tables (Markdown mit
Zell-Rects, Gerüst ungemappt — Raster wird als Vektorlinien
gezeichnet), other (TOC/bibref — raus aus main wie STREAM_OF).

RENDER-CANARY vergleicht die Extraktion gegen die tatsächlich
GEZEICHNETEN Tokens (Listen-Bullets und Notennummern stehen deshalb
auch im Strom-Text — nichts Gezeichnetes ist T0-fremd; Rasterlinien
sind Vektoren, keine Textzeichen).
"""
from __future__ import annotations

import re
from pathlib import Path

from enrich_core.dossier import Dossier, utc_now
from enrich_core.ids import new_id
from enrich_core.schemas.common import Agent
from enrich_core.schemas.intake import IntakeLayer
from enrich_core.schemas.layout import Block, LayoutLayer
from enrich_core.schemas.manifest import RunRecord, RunWarning
from enrich_core.schemas.text import (
    BlockSpan,
    DiffOp,
    NoteMarker,
    SpanRect,
    StreamDiff,
    TextCleanLayer,
    TextRawLayer,
)

from .schrift import _FONT_DIR, _sichtbar

TOOL = "enrich-textimport"
TOOL_VERSION = "0.1.0"
DETECTOR = "enrich-textsatz"

_SEITE = (595.28, 841.89)
_RAND = 70.0
_RAND_OBEN = 76.0
_RAND_UNTEN = 76.0
_ZEILE_FAKTOR = 1.5

_FACES = {
    "regular": "RecursiveSansLnrSt-Regular.ttf",
    "fett": "RecursiveSansLnrSt-Bold.ttf",
    "kursiv": "RecursiveSansLnrSt-Italic.ttf",
    "mono": "RecursiveMonoLnrSt-Regular.ttf",
}

#: Größe/Abstände je Blocktyp (Recursive-Mapping, User-Entscheid).
_TYP_STIL = {
    "title": (20.0, True, 14.0),
    "h1": (16.5, True, 12.0),
    "h2": (14.0, True, 10.0),
    "h3": (12.5, True, 8.0),
    "h4": (11.5, True, 7.0),
    "paragraph": (10.5, False, 6.0),
    "quote": (10.5, False, 6.0),
    "list-item": (10.5, False, 3.0),
    "toc": (10.0, False, 2.0),
    "bibref": (10.0, False, 4.0),
    "caption": (9.5, False, 5.0),
    "note": (9.5, False, 4.0),
    # Sprecher-Label-Zeile (2026-08-30): kompakt, dicht am Rede-Absatz
    "sprecher": (9.5, False, 1.0),
}

_STREAM_VON = {"title": "main", "h1": "main", "h2": "main",
               "h3": "main", "h4": "main", "paragraph": "main",
               "quote": "main", "list-item": "main",
               "caption": "captions", "toc": "other",
               "bibref": "other",
               "sprecher": "other"}
_BLOCKTYP = {"quote": "paragraph", "note": "footnote",
             "sprecher": "speaker-label"}


def _clamp_groesse(g: float) -> float:
    return max(7.0, min(26.0, g))


class _Setzer:
    """Der eine Durchlauf: setzt, misst und schreibt alle Karten."""

    def __init__(self) -> None:
        import fitz

        self.fitz = fitz
        self.fonts = {k: fitz.Font(fontfile=str(_FONT_DIR / v))
                      for k, v in _FACES.items()}
        self.doc = fitz.open()
        #: je Seite: {(face, farbe): TextWriter}
        self.schreiber: list[dict] = []
        self.linien: list[tuple[int, tuple]] = []  # (seiten_idx, rect)
        self.y = 0.0
        self.stream_text: dict[str, list[str]] = {}
        self.rects: list[SpanRect] = []
        self.bloecke: list[Block] = []
        self.block_spans: list[BlockSpan] = []
        self.note_markers: list[NoteMarker] = []
        self.marker_spans: list[tuple[int, int]] = []  # für T1-Ops
        self.gezeichnet: list[str] = []  # Canary-Referenz
        self.lesefolge = 0
        self._neue_seite()

    # ---------- Grundlagen ----------

    def _neue_seite(self) -> None:
        self.doc.new_page(width=_SEITE[0], height=_SEITE[1])
        self.schreiber.append({})
        self.y = _RAND_OBEN

    def _tw(self, face: str, farbe: str):
        seite = self.schreiber[-1]
        key = (face, farbe)
        if key not in seite:
            seite[key] = self.fitz.TextWriter(
                self.fitz.Rect(0, 0, *_SEITE))
        return seite[key]

    def _stream_pos(self, stream: str) -> int:
        teile = self.stream_text.setdefault(stream, [])
        return sum(len(t) for t in teile)

    def _anhaengen(self, stream: str, text: str) -> int:
        pos = self._stream_pos(stream)
        self.stream_text[stream].append(text)
        return pos

    def _wort(self, seiten_idx: int, x: float, y: float, text: str,
              face: str, size: float, farbe: str, stream: str,
              start: int, end: int, dy: float = 0.0) -> float:
        font = self.fonts[face]
        w = font.text_length(text, fontsize=size)
        self._tw(face, farbe).append((x, y - dy), text, font=font,
                                     fontsize=size)
        asc = font.ascender * size
        desc = abs(font.descender) * size
        bbox = (x, y - dy - asc, x + w, y - dy + desc)
        self.rects.append(SpanRect(stream=stream, start=start, end=end,
                                   page=seiten_idx + 1, bbox=bbox))
        self.gezeichnet.append(text)
        return w

    # ---------- Absätze ----------

    def absatz(self, typ: str, tokens: list[dict], stream: str,
               einzug: float = 0.0) -> None:
        """tokens: [{text, start, end, face, size, farbe, dy}] mit
        Strom-Offsets; text "" = reiner Abstandshalter."""
        size_basis, _fett, abstand = _TYP_STIL.get(
            typ, _TYP_STIL["paragraph"])
        zeilenhoehe = max(size_basis, max(
            (t["size"] for t in tokens), default=size_basis)) \
            * _ZEILE_FAKTOR
        breite = _SEITE[0] - 2 * _RAND - einzug
        x0 = _RAND + einzug
        if self.y + zeilenhoehe > _SEITE[1] - _RAND_UNTEN:
            self._schliesse_fragment(typ, stream)
            self._neue_seite()
        self.y += zeilenhoehe * 0.72  # Baseline der ersten Zeile
        x = x0
        self._frag = []
        self._frag_span: list[int] = []
        font_r = self.fonts["regular"]
        leer = font_r.text_length(" ", fontsize=size_basis)
        for t in tokens:
            font = self.fonts[t["face"]]
            w = font.text_length(t["text"], fontsize=t["size"])
            luecke = leer if t.get("space_davor") and x > x0 else 0.0
            if x + luecke + w > x0 + breite and x > x0:
                x = x0
                luecke = 0.0
                self.y += zeilenhoehe
                if self.y > _SEITE[1] - _RAND_UNTEN:
                    self._schliesse_fragment(typ, stream)
                    self._neue_seite()
                    self.y += zeilenhoehe * 0.72
            x += luecke
            w = self._wort(len(self.doc) - 1, x, self.y, t["text"],
                           t["face"], t["size"], t["farbe"], stream,
                           t["start"], t["end"], t.get("dy", 0.0))
            b = self.rects[-1].bbox
            self._frag.append(b)
            if not self._frag_span:
                self._frag_span = [t["start"], t["end"]]
            else:
                self._frag_span[1] = t["end"]
            x += w
        self._schliesse_fragment(typ, stream)
        self.y += zeilenhoehe * 0.28 + abstand

    def _schliesse_fragment(self, typ: str, stream: str) -> None:
        frag = getattr(self, "_frag", None)
        if not frag:
            return
        bid = new_id("b")
        btyp = _BLOCKTYP.get(typ, typ)
        self.bloecke.append(Block(
            id=bid, page=len(self.doc), type=btyp,  # type: ignore[arg-type]
            bbox=(min(r[0] for r in frag), min(r[1] for r in frag),
                  max(r[2] for r in frag), max(r[3] for r in frag)),
            reading_order=self.lesefolge))
        self.block_spans.append(BlockSpan(
            block_id=bid, stream=stream, start=self._frag_span[0],
            end=self._frag_span[1], page=len(self.doc), type=btyp))
        self.lesefolge += 1
        self._frag = []
        self._frag_span = []

    # ---------- Tabellen ----------

    def tabelle(self, zeilen: list[list[str]]) -> None:
        """Einfaches Raster gleicher Spaltenbreite; Zell-Wörter tragen
        tables-Strom-Offsets (Markdown-Gerüst ungemappt, text 0.2.0);
        Rasterlinien = Vektoren (Canary-neutral)."""
        cols = max(len(z) for z in zeilen)
        breite = _SEITE[0] - 2 * _RAND
        zell_b = breite / cols
        size = 9.5
        font = self.fonts["regular"]
        # tables-Strom: Markdown-Block
        md: list[str] = []
        vor = self._stream_pos("tables")
        if vor:
            self._anhaengen("tables", "\n\n")
        for zi, zeile in enumerate(zeilen):
            md_zeile = "|"
            for zelle in zeile + [""] * (cols - len(zeile)):
                md_zeile += f" {zelle} |"
            md.append(md_zeile + "\n")
            if zi == 0:
                md.append("|" + "---|" * cols + "\n")
        basis = self._anhaengen("tables", "".join(md))
        # Offsets nachjustieren (basis + relative Position)
        rel = 0
        zell_offsets = []
        for zi, zeile in enumerate(zeilen):
            offs = []
            rel += 1  # "|"
            for zelle in zeile + [""] * (cols - len(zeile)):
                rel += 1  # Space
                offs.append(basis + rel)
                rel += len(zelle) + 2  # Text + " |"
            zell_offsets.append(offs)
            rel += 1  # "\n"
            if zi == 0:
                rel += 1 + 4 * cols + 1  # |---|…\n
        # Rendern
        zeilenhoehe = size * _ZEILE_FAKTOR
        for zi, zeile in enumerate(zeilen):
            # Zeilenhöhe = max umbrochene Zellhöhe
            hoehen = []
            for zelle in zeile:
                worte = zelle.split()
                zl, x = 1, 0.0
                for wrt in worte:
                    w = font.text_length(wrt, fontsize=size)
                    if x + w > zell_b - 8 and x > 0:
                        zl += 1
                        x = 0.0
                    x += w + font.text_length(" ", fontsize=size)
                hoehen.append(zl)
            zh = max(hoehen or [1]) * zeilenhoehe + 6
            if self.y + zh > _SEITE[1] - _RAND_UNTEN:
                self._neue_seite()
            y_top = self.y
            frag: list[tuple] = []
            span: list[int] = []
            for ci, zelle in enumerate(zeile):
                zx = _RAND + ci * zell_b + 4
                zy = y_top + size * 1.1
                x = zx
                pos = zell_offsets[zi][ci]
                for wrt in zelle.split():
                    wrt_s = _sichtbar(font, wrt)
                    if not wrt_s:
                        pos += len(wrt) + 1
                        continue
                    w = font.text_length(wrt_s, fontsize=size)
                    if x + w > _RAND + (ci + 1) * zell_b - 4 \
                            and x > zx:
                        x = zx
                        zy += zeilenhoehe
                    self._wort(len(self.doc) - 1, x, zy, wrt_s,
                               "regular", size, "", "tables",
                               pos, pos + len(wrt))
                    frag.append(self.rects[-1].bbox)
                    if not span:
                        span = [pos, pos + len(wrt)]
                    else:
                        span[1] = pos + len(wrt)
                    x += w + font.text_length(" ", fontsize=size)
                    pos += len(wrt) + 1
            for ci in range(cols + 1):
                lx = _RAND + ci * zell_b
                self.linien.append((len(self.doc) - 1,
                                    (lx, y_top, lx, y_top + zh)))
            self.linien.append((len(self.doc) - 1,
                                (_RAND, y_top, _SEITE[0] - _RAND,
                                 y_top)))
            self.y = y_top + zh
            if frag and span:
                bid = new_id("b")
                self.bloecke.append(Block(
                    id=bid, page=len(self.doc), type="table",
                    bbox=(min(r[0] for r in frag),
                          min(r[1] for r in frag),
                          max(r[2] for r in frag),
                          max(r[3] for r in frag)),
                    reading_order=self.lesefolge))
                self.block_spans.append(BlockSpan(
                    block_id=bid, stream="tables", start=span[0],
                    end=span[1], page=len(self.doc), type="table"))
                self.lesefolge += 1
        self.linien.append((len(self.doc) - 1,
                            (_RAND, self.y, _SEITE[0] - _RAND,
                             self.y)))
        self.y += 10

    # ---------- Abschluss ----------

    def fertig(self) -> tuple[bytes, int]:
        for idx, seite in enumerate(self.schreiber):
            for (_face, farbe), tw in seite.items():
                rgb = None
                h = farbe.lstrip("#")
                if len(h) == 6:
                    rgb = tuple(int(h[i:i + 2], 16) / 255
                                for i in (0, 2, 4))
                tw.write_text(self.doc[idx], color=rgb)
        for idx, (x0, y0, x1, y1) in self.linien:
            self.doc[idx].draw_line((x0, y0), (x1, y1),
                                    color=(0.6, 0.6, 0.6), width=0.5)
        self.doc.set_metadata({
            "title": "", "author": "", "subject": "", "keywords": "",
            "creator": "enrich textimport", "producer": "enrich",
            "creationDate": "", "modDate": ""})
        try:
            self.doc.subset_fonts()
        except Exception:  # noqa: BLE001
            pass
        pdf = self.doc.tobytes(deflate=True, garbage=3)
        seiten = len(self.doc)
        self.doc.close()
        return pdf, seiten


def _face(fett: bool, kursiv: bool, mono: bool) -> str:
    if mono:
        return "mono"
    if fett:
        return "fett"
    if kursiv:
        return "kursiv"
    return "regular"


def setze_struktur(struktur: dict) -> dict:
    """Struktur (docx_lesen-Modell) → PDF + alle Schichten-Zutaten.

    Rückgabe: {pdf, seiten, streams, rects, bloecke, block_spans,
    note_markers, marker_ops, ranking, gezeichnet_canary}.
    """
    s = _Setzer()

    def absatz_setzen(typ: str, runs: list[dict],
                      noten: list[tuple[int, int, int]],
                      stream: str, prefix: str = "",
                      einzug: float = 0.0) -> tuple[int, int]:
        """Runs → Strom-Text + Tokens; gibt (start, ende) im Strom."""
        size_typ, fett_typ, _ = _TYP_STIL.get(typ,
                                              _TYP_STIL["paragraph"])
        if s._stream_pos(stream):
            s._anhaengen(stream, "\n")
        start = s._stream_pos(stream)
        noten_je_run: dict[int, list[tuple[int, int]]] = {}
        for ri, off, nr in noten:
            noten_je_run.setdefault(ri, []).append((off, nr))
        tokens: list[dict] = []
        text_teile: list[str] = []
        pos = start
        if prefix:
            s._anhaengen(stream, prefix)
            tokens.append({"text": prefix.strip(), "start": pos,
                           "end": pos + len(prefix.strip()),
                           "face": "regular", "size": size_typ,
                           "farbe": "", "space_davor": False})
            pos += len(prefix)
            text_teile.append(prefix)
        for ri in range(max(len(runs),
                            max(noten_je_run) + 1 if noten_je_run
                            else 0)):
            run = runs[ri] if ri < len(runs) else {
                "text": "", "fett": False, "kursiv": False,
                "farbe": "", "groesse": 0.0, "mono": False}
            face = _face(run["fett"] or fett_typ, run["kursiv"],
                         run["mono"])
            size = _clamp_groesse(run["groesse"]) \
                if run["groesse"] else size_typ
            marker = sorted(noten_je_run.get(ri, []))
            text = run["text"]
            stuecke: list[tuple[str, bool, int | None]] = []
            letzt = 0
            for off, nr in marker:
                stuecke.append((text[letzt:off], False, None))
                stuecke.append((str(nr), True, nr))
                letzt = off
            stuecke.append((text[letzt:], False, None))
            for stueck, hoch, _nr in stuecke:
                if not stueck:
                    continue
                if hoch:
                    p0 = pos
                    s._anhaengen(stream, stueck)
                    s.note_markers.append(NoteMarker(
                        stream=stream, start=p0,
                        end=p0 + len(stueck), number=_nr))
                    if stream == "main":
                        s.marker_spans.append((p0, p0 + len(stueck)))
                    tokens.append({
                        "text": stueck, "start": p0,
                        "end": p0 + len(stueck), "face": face,
                        "size": size * 0.65, "farbe": run["farbe"],
                        "dy": size * 0.35, "space_davor": False})
                    pos += len(stueck)
                    text_teile.append(stueck)
                    continue
                # Wort-Tokens mit Lücken-Wissen
                for m in re.finditer(r"\S+", stueck):
                    wort = _sichtbar(s.fonts[face], m.group())
                    davor = stueck[:m.start()]
                    space = bool(davor and davor[-1].isspace()) \
                        or (not davor and text_teile
                            and text_teile[-1][-1:].isspace())
                    p0 = pos + m.start()
                    if wort:
                        tokens.append({
                            "text": wort, "start": p0,
                            "end": p0 + len(m.group()), "face": face,
                            "size": size, "farbe": run["farbe"],
                            "space_davor": space})
                s._anhaengen(stream, stueck)
                pos += len(stueck)
                text_teile.append(stueck)
        ende = s._stream_pos(stream)
        s.absatz(typ, tokens, stream, einzug=einzug)
        return start, ende

    for el in struktur["absaetze"]:
        if el["typ"] == "tabelle":
            s.tabelle(el["zeilen"])
            continue
        typ = el["typ"]
        stream = _STREAM_VON.get(typ, "main")
        prefix = "• " if typ == "list-item" else ""
        einzug = 24.0 if typ == "quote" else (
            14.0 if typ == "list-item" else 0.0)
        absatz_setzen(typ, el["runs"], el.get("noten", []),
                      stream, prefix=prefix, einzug=einzug)

    noten_alle = ([("footnote", nr, txt)
                   for nr, txt in struktur.get("fussnoten", [])]
                  + [("endnote", nr, txt)
                     for nr, txt in struktur.get("endnoten", [])])
    if noten_alle:
        s._neue_seite()
        # Überschrift in den APPARAT-Strom — main bleibt frei von
        # Render-Artefakten (Analyse-Text ist heilig)
        absatz_setzen("h2", [{"text": "Noten", "fett": True,
                              "kursiv": False, "farbe": "",
                              "groesse": 0.0, "mono": False}],
                      [], "footnotes")
        for _art, nr, txt in sorted(noten_alle, key=lambda n: n[1]):
            absatz_setzen("note", [{"text": txt, "fett": False,
                                    "kursiv": False, "farbe": "",
                                    "groesse": 0.0, "mono": False}],
                          [], "footnotes", prefix=f"{nr} ")

    pdf, seiten = s.fertig()
    streams = {name: "".join(teile)
               for name, teile in s.stream_text.items() if teile}
    # T1: main ohne Marker-Ziffern (note-marker-Ops)
    marker_ops = [DiffOp(op="note-marker", t0=(a, b), t1=(0, 0))
                  for a, b in sorted(s.marker_spans)]
    ranking = [_TYP_STIL[f"h{i}"][0] for i in range(1, 5)]
    return {"pdf": pdf, "seiten": seiten, "streams": streams,
            "rects": s.rects, "bloecke": s.bloecke,
            "block_spans": s.block_spans,
            "note_markers": s.note_markers, "marker_ops": marker_ops,
            "ranking": ranking,
            "gezeichnet": "".join(s.gezeichnet)}


def canary(pdf: bytes, gezeichnet: str) -> int:
    """Extraktion gegen die gezeichneten Tokens — als ZEICHEN-
    MULTIMENGE: der Content-Stream ist nach (Schnitt, Farbe)
    gruppiert (ein TextWriter je Kombination), seine Reihenfolge ist
    also nie Lesereihenfolge. Der Canary prüft, was er prüfen soll —
    Glyphen-Treue (Emoji/ZWSP/Clipping ändern die Multimenge); die
    POSITIONEN garantieren wir selbst per Konstruktion (SpanRects)."""
    from collections import Counter

    import fitz

    doc = fitz.open(stream=pdf, filetype="pdf")
    ext = Counter("".join(w[4] for p in doc
                          for w in p.get_text("words")))
    doc.close()
    soll = Counter(re.sub(r"\s+", "", gezeichnet))
    diff = (ext - soll) + (soll - ext)
    return sum(diff.values())


def baue_struktur_dossier(pfad: Path, struktur: dict, *, quelle: str,
                          user: str,
                          zeiten: list[dict] | None = None,
                          audio: Path | None = None,
                          zeiten_quelle: str = "") -> tuple[Dossier,
                                                            dict]:
    """Vollständiges Dossier aus einer Struktur; optional Zeitkarte
    (Transkript) + Audio-Kopie. Gibt (Dossier, bericht)."""
    import shutil
    import tempfile

    from enrich_core.canonical import content_hash
    from enrich_core.diffmap import project_span
    from enrich_core.schemas.zeitkarte import ZeitEinheit, ZeitkarteLayer

    satz = setze_struktur(struktur)
    can = canary(satz["pdf"], satz["gezeichnet"])
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tf:
        tf.write(satz["pdf"])
        tf.flush()
        d = Dossier.create(pfad, source_pdf=Path(tf.name))
    agent = Agent(type="derived", tool=f"{TOOL}/{TOOL_VERSION}")
    warnungen: list[RunWarning] = []
    if can:
        warnungen.append(RunWarning(
            code="W-REFI-RENDER",
            message=f"Render-Canary: {can} Zeichen-Abweichungen"))

    def lauf(layer_name: str, extra: dict) -> RunRecord:
        return RunRecord(id=new_id("run"), tool=TOOL,
                         tool_version=TOOL_VERSION, layer=layer_name,
                         agent=agent, inputs={},
                         config={"quelle": quelle,
                                 "renderer": DETECTOR},
                         started=utc_now(), warnings=warnungen,
                         summary=extra)

    d.write_layer("0-intake.json", IntakeLayer(
        agent=agent, verdict="accept", doc_class="born-digital",
        page_count=satz["seiten"], text_source="embedded"),
        lauf("0-intake.json", {"quelle": quelle}))
    d.write_layer("1-layout.json", LayoutLayer(
        agent=agent, detector=DETECTOR, page_count=satz["seiten"],
        heading_size_ranking=satz["ranking"],
        blocks=satz["bloecke"]),
        lauf("1-layout.json", {"bloecke": len(satz["bloecke"])}))
    d.write_layer("2-text-raw.json", TextRawLayer(
        agent=agent, streams=satz["streams"], spans=satz["rects"],
        block_spans=satz["block_spans"],
        note_markers=satz["note_markers"]),
        lauf("2-text-raw.json",
             {"zeichen": sum(len(t) for t in satz["streams"].values()),
              "woerter": len(satz["rects"])}))
    # T1: main ohne Marker-Ziffern, andere Ströme identisch
    t1_streams: dict[str, str] = {}
    diffmaps: dict[str, StreamDiff] = {}
    hashes: dict[str, str] = {}
    for name, text in satz["streams"].items():
        if name == "main" and satz["marker_ops"]:
            t1 = []
            letzt = 0
            ops = []
            entfernt = 0
            for op in satz["marker_ops"]:
                a, b = op.t0
                t1.append(text[letzt:a])
                ziel = a - entfernt
                ops.append(DiffOp(op="note-marker", t0=(a, b),
                                  t1=(ziel, ziel)))
                entfernt += b - a
                letzt = b
            t1.append(text[letzt:])
            t1_streams[name] = "".join(t1)
            diffmaps[name] = StreamDiff(ops=ops)
        else:
            t1_streams[name] = text
            diffmaps[name] = StreamDiff(ops=[])
        hashes[name] = content_hash(t1_streams[name])
    d.write_layer("3-text-clean.json", TextCleanLayer(
        revision="r1", agent=agent, streams=t1_streams,
        diffmaps=diffmaps, stream_hashes=hashes),
        lauf("3-text-clean.json", {"canary_fehler": can}))

    bericht = {"seiten": satz["seiten"], "canary": can,
               "bloecke": len(satz["bloecke"]),
               "streams": {k: len(v) for k, v in t1_streams.items()},
               "noten": len(satz["note_markers"]), "audio": "",
               "turns": 0}

    if zeiten:
        # Offsets kommen als T0-main — auf T1 projizieren
        ops = diffmaps["main"].ops
        einheiten = []
        for z in zeiten:
            pr = project_span(ops, z["start"], z["end"],
                              direction="t0->t1")
            einheiten.append(ZeitEinheit(
                start=pr.start, end=pr.end, t0_s=z["t0_s"],
                t1_s=z["t1_s"], speaker=z.get("speaker", "")))
        audio_name = ""
        if audio is not None and audio.is_file():
            audio_name = f"audio{audio.suffix.lower()}"
            shutil.copyfile(audio, d.path / audio_name)
            bericht["audio"] = audio_name
        d.write_layer("2z-zeitkarte.json", ZeitkarteLayer(
            agent=agent, revision="r1", audio=audio_name,
            quelle=zeiten_quelle, einheiten=einheiten),
            lauf("2z-zeitkarte.json", {"turns": len(einheiten)}))
        bericht["turns"] = len(einheiten)

    note = "Text-Import: deterministisch gesetzt, nichts zu prüfen"
    d.set_gate("layout", user, note=note)
    d.set_gate("ocr", user,
               note="Text-Import: kein OCR — der Text ist das Original")
    return d, bericht
