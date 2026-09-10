#!/usr/bin/env python3
"""Die Markdown-Textbausteine aus docs/ in index.html einbetten.

Warum: Der Knopf «… (.md)» soll die Datei SPEICHERN, nicht anzeigen.
Das download-Attribut tut das nur, wenn die Seite von einem Server
kommt — von file:// öffnen Chromium und WebKit die .md stattdessen im
Fenster (geprüft 2026-09-10). Ein Blob aus eingebettetem Text lädt
überall herunter. Die Dateien in docs/ bleiben die Quelle; nach jeder
Änderung daran dieses Skript laufen lassen:

    python3 site/docs/einbetten.py
"""
from pathlib import Path
import re

HIER = Path(__file__).resolve().parent
INDEX = HIER.parent / "index.html"
DATEIEN = {
    "verfahren": {"de": "localtranscript-datenverarbeitung-de.md",
                  "en": "localtranscript-data-processing-en.md",
                  "fr": "localtranscript-traitement-des-donnees-fr.md",
                  "it": "localtranscript-trattamento-dei-dati-it.md"},
    "blatt":     {"de": "localtranscript-app-blatt-de.md",
                  "en": "localtranscript-fact-sheet-en.md",
                  "fr": "localtranscript-fiche-fr.md",
                  "it": "localtranscript-scheda-it.md"},
}
ANFANG, ENDE = "<!-- docs:start -->", "<!-- docs:end -->"

bloecke = []
for art, je_sprache in DATEIEN.items():
    for lang, name in je_sprache.items():
        text = (HIER / name).read_text(encoding="utf-8")
        # Ein «</script» im Text würde den Block beenden — entschärfen.
        text = text.replace("</script", "<\\/script")
        bloecke.append(f'<script type="text/plain" data-md="{art}-{lang}" '
                       f'data-name="{name}">\n{text}</script>')
neu = ANFANG + "\n" + "\n".join(bloecke) + "\n" + ENDE

s = INDEX.read_text(encoding="utf-8")
if ANFANG in s:
    s = re.sub(re.escape(ANFANG) + r".*?" + re.escape(ENDE), lambda m: neu, s, flags=re.S)
else:
    s = s.replace("\n<script>\nconst T = {", "\n" + neu + "\n\n<script>\nconst T = {", 1)
INDEX.write_text(s, encoding="utf-8")
print(f"{len(bloecke)} Textbausteine eingebettet ({sum(len(b) for b in bloecke)//1024} KB)")
