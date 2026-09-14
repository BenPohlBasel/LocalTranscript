"""Drift-Guard: den gevendorten enrich-Baustein (turns.py, reine
Transkript-Funktionen aus textimport) gegen die lokale enrich-Quelle halten (Muster enrich/praedikate-Worker). Läuft nur,
wenn ../enrich auf dieser Maschine liegt."""
from __future__ import annotations

from pathlib import Path

import pytest

ENRICH = Path(__file__).resolve().parents[2].parent / "enrich" \
    / "packages" / "enrich-serve" / "src" / "enrich_serve"
VENDOR = Path(__file__).resolve().parents[1] / "src" \
    / "turnscript" / "enrich_export"

pytestmark = pytest.mark.skipif(not ENRICH.is_dir(),
                                reason="enrich-Checkout fehlt")


def _ohne_kopf(text: str) -> str:
    zeilen = [z for z in text.splitlines()
              if not z.startswith("# VENDORED")
              and not z.startswith("# Stand enrich@")]
    while zeilen and zeilen[0].startswith("#"):
        zeilen.pop(0)
    return "\n".join(zeilen) + "\n"


def test_turns_drift():
    quelle = (ENRICH / "textimport.py").read_text()
    vendor = (VENDOR / "turns.py").read_text()
    kern = vendor.split("import re\n", 1)[1].lstrip("\n")
    assert kern in quelle, (
        "turns.py (pure Transkript-Funktionen) ist gegen "
        "enrich/textimport.py gedriftet")
