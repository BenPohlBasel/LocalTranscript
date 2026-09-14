"""Zotero-Metadaten für ein Transkript — über enrich_core.zotero.

Der Weg ist der von enrich (User-Entscheid dort 2026-07-28): nicht die
Zotero-API, sondern die lokale `zotero.sqlite`, geöffnet mit
`?immutable=1` — lesend, ohne Sperre, Zotero merkt nichts. Freigeschaltet
nur durch `zotero_consent` in den Einstellungen; ohne sie 424.

Ein Transkript hat kein PDF zum Hashen — Kandidaten kommen über die
Nähe von Titel, Jahr, Citekey und Namen; der Zotero-Typ «Interview»
wird bevorzugt. Was ins Transkript übernommen wird, entscheidet die
Person je Creator-Rolle (Pseudonymisierung: Interviewer:in ja, die
befragte Person nur, wenn das Transkript nicht pseudonymisiert ist oder
Zotero selbst schon das Pseudonym führt).
"""
from __future__ import annotations

import re
import unicodedata

from enrich_core import zotero as ez
from enrich_core.errors import PreconditionError

from .config import read_config

#: Rollen, die Zotero für Interviews kennt — Anzeige-Reihenfolge
ROLLEN = ("interviewer", "interviewee", "author", "contributor",
          "editor", "translator")


class ZoteroFehler(Exception):
    """Zugriff nicht möglich — Einwilligung fehlt oder keine Datenbank."""


def status() -> dict:
    """Ohne Einwilligung wird auch nicht GESUCHT (die Suche liest Zoteros
    prefs.js) — `found` bleibt dann unbekannt (None)."""
    cfg = read_config()
    if not cfg.get("zotero_consent"):
        return {"consent": False, "found": None, "dir": None}
    d = ez.find_zotero_dir(cfg)
    return {"consent": True, "found": d is not None,
            "dir": str(d) if d else None}


def _zugang():
    try:
        return ez.zotero_access(read_config())
    except PreconditionError as e:
        raise ZoteroFehler(str(e)) from e


def _tokens(text: str) -> set[str]:
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 2}


def _kurz(it: dict) -> dict:
    return {"item_key": it["item_key"], "title": it.get("title"),
            "year": it.get("year"), "date": datum(it.get("date")),
            "item_type": it.get("item_type"), "citekey": it.get("citekey"),
            "creators": it.get("creators") or [],
            "publication": it.get("publication"),
            "library": it.get("library"),
            "select_link": it.get("select_link")}


def kandidaten(q: str, limit: int = 12) -> list[dict]:
    """Die besten Treffer für eine Suchbasis (Name des Transkripts oder
    freie Eingabe). Interviews zuerst, dann nach Nähe."""
    zdir = _zugang()
    items = ez.items_with_pdfs(zotero_dir=zdir, mit_pdf=False)
    qt = _tokens(q)
    aus = []
    for it in items:
        it_t = _tokens(" ".join(filter(None, [
            str(it.get("title") or ""), str(it.get("year") or ""),
            str(it.get("citekey") or ""),
            " ".join(f"{c.get('first', '')} {c.get('last', '')}"
                     for c in it.get("creators") or [])])))
        score = (len(qt & it_t) / len(qt)) if qt else 0.0
        interview = (it.get("item_type") or "") == "interview"
        if score <= 0.0 and not (interview and not qt):
            continue
        k = _kurz(it)
        k["score"] = round(score + (0.25 if interview else 0.0), 3)
        aus.append(k)
    aus.sort(key=lambda c: (-c["score"], c.get("year") or "", c.get("title") or ""))
    return aus[:limit]


def eintrag(item_key: str) -> dict:
    zdir = _zugang()
    for it in ez.items_with_pdfs(zotero_dir=zdir, mit_pdf=False):
        if it["item_key"] == item_key:
            return it
    raise ZoteroFehler(f"Zotero-Eintrag {item_key} nicht gefunden")


_ZDATUM = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:\s|$)")


def datum(roh: str | None) -> str | None:
    """Zoteros `date`-Feld → lesbares Datum. Zotero speichert
    «YYYY-MM-DD Original» (Sortierform, dann die Eingabe: «2006-04-01
    2006-04-01», «2019-00-00 2019»); unbekannte Teile sind 00 und
    fallen weg. Was nicht in die Form passt, bleibt, wie es ist."""
    if not roh:
        return None
    m = _ZDATUM.match(roh.strip())
    if not m:
        return roh.strip()
    teile = [t for t in m.groups() if t != "00" and t != "0000"]
    return "-".join(teile) if teile else roh.strip()


def schnappschuss(it: dict, rollen: list[str]) -> dict:
    """Was ins Transkript geht: der Eintrag ohne Personen, deren Rolle
    nicht gewählt wurde. `origin: source` — unverändert aus Zotero."""
    gewaehlt = set(rollen)
    return {"item_key": it["item_key"], "citekey": it.get("citekey"),
            "item_type": it.get("item_type"), "title": it.get("title"),
            "date": datum(it.get("date")), "year": it.get("year"),
            "publication": it.get("publication"), "doi": it.get("doi"),
            "abstract": it.get("abstract"),
            "select_link": it.get("select_link"),
            "creators": [{"first": c.get("first", ""), "last": c.get("last", ""),
                          "role": c.get("role", "author")}
                         for c in it.get("creators") or []
                         if c.get("role", "author") in gewaehlt],
            "rollen": sorted(gewaehlt), "origin": "source"}
