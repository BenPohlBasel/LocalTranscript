"""Drift-Guard: die gevendorten enrich-Bausteine gegen die lokale
enrich-Quelle halten (Muster enrich/praedikate-Worker). Läuft nur,
wenn ../enrich auf dieser Maschine liegt."""
from __future__ import annotations

from pathlib import Path

import pytest

ENRICH = Path(__file__).resolve().parents[2].parent / "enrich" \
    / "packages" / "enrich-serve" / "src" / "enrich_serve"
VENDOR = Path(__file__).resolve().parents[1] / "src" \
    / "localtranscript" / "enrich_export"

pytestmark = pytest.mark.skipif(not ENRICH.is_dir(),
                                reason="enrich-Checkout fehlt")


def _ohne_kopf(text: str) -> str:
    zeilen = [z for z in text.splitlines()
              if not z.startswith("# VENDORED")
              and not z.startswith("# Stand enrich@")]
    while zeilen and zeilen[0].startswith("#"):
        zeilen.pop(0)
    return "\n".join(zeilen) + "\n"


def test_textsatz_drift():
    quelle = (ENRICH / "textsatz.py").read_text()
    vendor = (VENDOR / "textsatz.py").read_text()
    quelle = quelle.replace(
        "from .refi_text import _FONT_DIR, _sichtbar",
        "from .schrift import _FONT_DIR, _sichtbar")
    assert _ohne_kopf(vendor) == quelle, (
        "textsatz.py ist gegen enrich gedriftet — neu vendoren "
        "(cp + Import-Patch, s. Kopfkommentar)")


def test_turns_drift():
    quelle = (ENRICH / "textimport.py").read_text()
    vendor = (VENDOR / "turns.py").read_text()
    kern = vendor.split("import re\n", 1)[1].lstrip("\n")
    assert kern in quelle, (
        "turns.py (pure Transkript-Funktionen) ist gegen "
        "enrich/textimport.py gedriftet")


def test_sichtbar_drift():
    quelle = (ENRICH / "refi_text.py").read_text()
    vendor = (VENDOR / "schrift.py").read_text()
    kern = vendor.split("def _sichtbar", 1)[1]
    kern_zeilen = [z for z in kern.splitlines() if z.strip()
                   and not z.strip().startswith(('"""', "Format",
                                                 "weicht", "Zeichen"))]
    for zeile in kern_zeilen:
        if zeile.strip().startswith("#"):
            continue
        assert zeile in quelle, f"schrift.py driftet bei: {zeile!r}"
