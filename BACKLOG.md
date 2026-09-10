# Backlog — LocalTranscript

Offene Aufgaben, die nicht aus dem Code oder der Git-Historie
hervorgehen. Erledigtes wandert ins CHANGELOG.

## Offen (nach Priorität)

1. **`origin` je Segment und je Sprecher in `transkript.json` (Auftrag
   2026-09-10, Format 2 des enrich-Dossiers — `../enrich/FORMAT.md` §0.1
   und §5).** Whisper schreibt `machine`; jedes Segment, das eine Person
   im Editor anfasst (Text, Sprecher, Grenzen, Teilen/Verbinden), wird
   `human` und behält den ersetzten Stand als `supersedes`; ein Sprecher
   wird `human`, sobald benannt. `human` gewinnt und ist nie ableitbar:
   nur der Editor setzt es, nie ein Import oder ein Lauf. Der Kopf der
   Datei nach dem Schicht-Vertrag: `schema: transcript/1.0.0`, `kind`,
   `origin: mixed`, `by` (Whisper-Modell, Diarisierung, App-Version),
   `did` (ein Satz). Kanonisches JSON (sortierte Schlüssel, `\n` am
   Ende) — heute `indent=2`, nicht hash-stabil. Der enrich-Export legt
   die Schicht als `source/transcript.json` mit Hash ins Manifest
   (`files`, `layers`), die Zeitkarte wird daraus abgeleitet.
   Bibliothek: Lesen alter `transkript.json` ohne `origin` → alle
   Segmente `machine`, Sprecher mit Name ≠ «Sprecher n» → `human`.
   Tests: Rundlauf hält die Flags; Import aus einem Dossier ebnet sie
   nicht ein.

2. **Journal im Transkript (`../enrich/FORMAT.md` §3.1).** Jede
   Schreibung im Editor ist ein Run mit `origin: human`, `who`
   (App-Kennung `localtranscript/2.2.0` immer; Person nur, wenn in den
   Einstellungen eine E-Mail steht — unbekannt ist kein Fehler), `started`/`finished`, `changed` (Segment-/Sprecher-
   IDs mit Art der Änderung), je Sitzung gebündelt (10 min Ruhe oder
   Verlassen des Editors). Der Whisper-Lauf ist ein Run mit `origin:
   machine` (Modell, Diarisierung, Version). Die `history/`-
   Schnappschüsse bleiben und werden vom Run referenziert. Beim
   enrich-Export gehen die Runs ins Manifest. Klein halten: IDs und
   Hashes, nie Texte oder Diffs; Richtwert unter 2 KB je Run.

3. **Import-Dialog für Ordner-Dossiers ohne enrich.app.** Ohne die UTI
   (kommt mit dem nächsten Build über `src-tauri/Info.plist`) ist ein
   `.enrich`-Verzeichnis im Dateidialog nicht wählbar; Drag & Drop
   geht. Nach dem Build prüfen, ob der Dialog das Package anbietet;
   sonst zweiter Knopf «Ordner wählen …» (`pickOrdner` gibt es).

4. **Anleitung, Schritt Exportieren:** Satz zum `.qdpx.zip` — vor dem
   Import in ATLAS.ti entpacken; `.qdpx` und `Media`-Ordner müssen
   nebeneinander liegen (User 2026-09-10). Viersprachig, Seite und
   App-Blatt.
