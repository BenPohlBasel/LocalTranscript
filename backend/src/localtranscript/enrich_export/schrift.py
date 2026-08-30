# VENDORED-Extrakt aus enrich@3d2b131 —
# packages/enrich-serve/src/enrich_serve/refi_text.py (_FONT_DIR,
# _GLYPH_CACHE, _sichtbar; _FONT_DIR zeigt hier auf die mitgelieferten
# Recursive-Fonts, OFL 1.1 — fonts/LICENSE-OFL.txt).
from __future__ import annotations

from pathlib import Path

_FONT_DIR = Path(__file__).parent / "fonts"

_GLYPH_CACHE: dict[int, bool | str] = {}


def _sichtbar(font, wort: str) -> str:
    """Render-Normalisierung (T1 bleibt IMMER byte-treu, nur das BILD
    weicht kontrolliert ab; der Canary nutzt DIESELBE Regel):
    Format-Zeichen (Unicode Cf — ZWSP/ZWJ/BOM) fallen ganz; sichtbare
    Zeichen ohne Glyph (Emoji, U+FFFD) werden "?"."""
    import unicodedata

    aus = []
    for ch in wort:
        cp = ord(ch)
        ok = _GLYPH_CACHE.get(cp)
        if ok is None:
            if unicodedata.category(ch) == "Cf":
                ok = "weg"
            else:
                ok = cp != 0xFFFD and bool(font.has_glyph(cp))
            _GLYPH_CACHE[cp] = ok
        if ok == "weg":
            continue
        aus.append(ch if ok else "?")
    return "".join(aus)
