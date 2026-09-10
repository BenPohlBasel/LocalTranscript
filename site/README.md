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

`img/<motiv>-<sprache>.png`, acht Motive mal vier Sprachen. Beim
Sprachwechsel tauscht das Skript die Quelle jedes `img[data-img]` aus,
die Seite zeigt die App also immer in der gerade gewählten Sprache.

Sie stammen aus einem echten Lauf in einer Wegwerf-Bibliothek, mit dem
**erfundenen** Interview aus `docs/demo/` (macOS `say`, zwei Stimmen,
2:02 min). Kein echtes Forschungsmaterial.

Neu aufnehmen: App in der gewünschten Sprache starten, Motive schießen,
mit Pillow auf Breite bringen und als PNG-8 sichern (256 Farben ohne
Dithering — UI-Flächen sind flach, das spart zwei Drittel, ohne dass die
Schriftkanten leiden).

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
