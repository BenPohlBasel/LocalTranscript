#!/usr/bin/env python3
"""Textbausteine aus docs/ als .docx erzeugen und beides in index.html
einbetten.

Warum einbetten: Der Knopf soll die Datei SPEICHERN, nicht anzeigen.
Das download-Attribut tut das nur, wenn die Seite von einem Server
kommt — von file:// öffnen Chromium und WebKit die .md im Fenster
(geprüft 2026-09-10). Ein Blob aus eingebettetem Inhalt lädt überall
herunter; die .docx liegt dafür als Base64 in der Seite (8 × 14 KB).

Die .md in docs/ sind die Quelle. Nach jeder Änderung daran:

    python3 site/docs/einbetten.py

Das Skript erzeugt die .docx mit pandoc (falls vorhanden) neu und
spielt .md und .docx in index.html ein. Die Seite selbst braucht
keinen Build-Schritt — nur dieser Ordner hat einen.
"""
from pathlib import Path
import base64, re, shutil, subprocess

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

pandoc = shutil.which("pandoc")
if not pandoc:
    print("pandoc fehlt — .docx werden nicht neu erzeugt, vorhandene eingebettet")

bloecke = []
for art, je_sprache in DATEIEN.items():
    for lang, name in je_sprache.items():
        md = HIER / name
        docx = md.with_suffix(".docx")
        if pandoc:
            subprocess.run([pandoc, str(md), "-o", str(docx),
                            "--from", "gfm", "--to", "docx"], check=True)
        text = md.read_text(encoding="utf-8").replace("</script", "<\\/script")
        bloecke.append(f'<script type="text/plain" data-md="{art}-{lang}" '
                       f'data-name="{name}">\n{text}</script>')
        if docx.exists():
            b64 = base64.b64encode(docx.read_bytes()).decode("ascii")
            bloecke.append(f'<script type="text/plain" data-docx="{art}-{lang}" '
                           f'data-name="{docx.name}">{b64}</script>')

neu = ANFANG + "\n" + "\n".join(bloecke) + "\n" + ENDE
s = INDEX.read_text(encoding="utf-8")
if ANFANG in s:
    s = re.sub(re.escape(ANFANG) + r".*?" + re.escape(ENDE), lambda m: neu, s, flags=re.S)
else:
    s = s.replace("\n<script>\nconst T = {", "\n" + neu + "\n\n<script>\nconst T = {", 1)
INDEX.write_text(s, encoding="utf-8")
kb = sum(len(b) for b in bloecke) // 1024
print(f"{len(bloecke)} Blöcke eingebettet ({kb} KB)")
