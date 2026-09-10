# LocalTranscript — statische Website

Ein Ordner, drei Sachen darin, sonst nichts. Zum Veröffentlichen den
Inhalt auf einen beliebigen Webserver legen; zum Anschauen `index.html`
doppelklicken — die Seite läuft auch von `file://`.

```
index.html          alles: Auszeichnung, Stil, Übersetzungen
fonts/              Recursive Variable (woff2) + OFL-Lizenztext
img/                Bildschirmfotos, je Motiv in vier Sprachen
```

## Was die Seite bewusst NICHT tut

- **Keine Cookies, kein Storage.** Kein `document.cookie`, kein
  `localStorage`, keine Messung, kein Zählpixel.
- **Keine fremden Server.** Die Schrift liegt im Ordner, nicht bei
  Google Fonts. Kein CDN, kein Framework, kein Build-Schritt. Die
  einzigen ausgehenden Adressen sind die Links, die der Besucher
  anklickt: GitHub und LinkedIn.
- **Keine Videos.** Nur Bildschirmfotos aus der laufenden App.

## Sprachen

Deutsch, Englisch, Französisch, Italienisch — dieselben vier wie in der
App. Umgeschaltet wird oben rechts; die Wahl landet in `?lang=xx`, damit
ein geteilter Link sie mitnimmt. Der **Hash bleibt den Abschnitten**
(`#guide`, `#dpo`, `#compare`), sonst zerschösse ein Sprachwechsel jeden
Link auf einen Abschnitt.

Ohne `?lang=` entscheidet die Browsersprache, sonst Deutsch.

Alle Texte stehen im Objekt `T` am Ende von `index.html`, ein Block je
Sprache, gleiche Schlüssel. Ein Schlüssel, der in einer Sprache fehlt,
lässt dort die deutsche Fassung stehen — besser kein Loch als eine leere
Zeile.

## Bildschirmfotos

`img/<motiv>-<sprache>.png`, acht Motive mal vier Sprachen — das
Hero-Bild und zu jedem Schritt der Anleitung eines. Die Anleitungsbilder
stammen aus einem auf 800 × 500 verkleinerten Fenster: neben dem Text
bleibt so alles lesbar, was in einem 1280er-Fenster zu Fliegendreck
würde (Vorschlag des Users). Beim
Sprachwechsel tauscht das Skript die Quelle jedes `img[data-img]` aus,
die Seite zeigt die App also immer in der gerade gewählten Sprache.

Sie stammen aus einem echten Lauf in einer Wegwerf-Bibliothek, mit dem
**erfundenen** Interview aus `docs/demo/` (macOS `say`, zwei Stimmen,
2:02 min). Kein echtes Forschungsmaterial.

Neu aufnehmen: App in der gewünschten Sprache starten, Motive schießen,
mit Pillow auf Breite bringen und als PNG-8 sichern (256 Farben ohne
Dithering — UI-Flächen sind flach, das spart zwei Drittel, ohne dass die
Schriftkanten leiden).

## Textbausteine (docs/)

Zwei Textbausteine je Sprache, als Markdown und als Word-Datei, im
Abschnitt «Texte für Datenschutz und Methodenteil» in einem
Vorschaufenster lesbar und von dort zu speichern. Die Knöpfe bauen die
Datei als Blob aus Inhalt, der in `index.html` eingebettet ist (die .md
als Klartext, die .docx als Base64, zusammen 190 KB) — das
`download`-Attribut allein tut das nur von einem Server aus; von
`file://` öffnen Chromium und WebKit die Datei im Fenster.

Die `.md` in `docs/` bleiben die Quelle. Nach jeder Änderung daran:

    python3 site/docs/einbetten.py

Das erzeugt die `.docx` mit pandoc neu und spielt beides in
`index.html` ein. Ohne diesen Lauf lädt der Knopf den alten Stand.

- **Verfahrensbeschreibung** — in Sachform, zehn Abschnitte nach dem
  Muster eines Verfahrensverzeichnisses, mit `[eckigen Klammern]` für
  das, was nur die verantwortliche Stelle weiss. Zum Einfügen in
  Verfahrensverzeichnis, DSFA, Ethikantrag, Datenmanagementplan.
- **App-Blatt für Forschende** — welche KI was tut (Tabelle), wo der
  Code liegt, warum nichts den Rechner verlässt, wie Namen ersetzt
  werden, plus ein fertiger Satz für den Methodenteil.

Beide sagen ausdrücklich, was NICHT abgedeckt ist: Backups und
Diagnosedaten von macOS, und dass Exporte mit Audio (.qdpx, .enrich.zip)
die Aufnahme samt Stimme und Klarnamen tragen.

## Farbe

Ein Ton, in Stufen: `--lila-600` (#4f46e5) trägt Knöpfe und Links,
`--lila-700` den Zeigezustand, `--lila-300` die Links auf dunklem Grund,
`--lila-050` die eigene Spalte der Vergleichstabelle. Der Ton kommt aus
dem App-Zeichen (#6366f1) — Seite und Programm tragen dieselbe Farbe.

## Zeichen

Das Zeichen ist die Initiale der Überschrift — der Name steht als
Subjekt im Satz, in derselben Schrift wie der Rest. Kopfzeile und
Favicon tragen dasselbe Zeichen; es ist das App-Symbol
App-Symbol (`frontend/src-tauri/icons/`), nur als Inline-SVG
nachgezeichnet: drei Balken auf lila Quadrat, `#6366f1`. Kein
zusätzlicher Abruf, scharf in jeder Grösse. Gegen das PNG geprüft —
die Abweichung liegt bei 2,4 von 255 und steckt in den Kantenglättungen.

## Schrift

Recursive Variable Font, **SIL OFL 1.1** — Lizenztext in
`fonts/OFL.txt`. Die Nennung bei jeder Veröffentlichung ist Pflicht und
steht in der Fußzeile der Seite sowie hier.

## Inhaltliche Vorbehalte

Die Vergleichstabelle nennt Preise und Eigenschaften fremder Programme
nach Herstellerangaben, Stand September 2026. Sie ist nach bestem Wissen
zusammengestellt, aber nicht nachgeprüft; die Fußnote unter der Tabelle
sagt das auch den Besuchern. Vor dem Veröffentlichen lohnt ein Blick auf
die aktuellen Preisseiten.
