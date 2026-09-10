# Backlog — LocalTranscript

Offene Aufgaben, die nicht aus dem Code oder der Git-Historie
hervorgehen. Erledigtes wandert ins CHANGELOG.

## Offen (nach Priorität)

1. ~~**`origin` je Segment und je Sprecher**~~ — GEBAUT 2026-09-10/11
   (Schema 2 der Bibliothek, `transcript.json` als Quelle im Container;
   CHANGELOG «Unreleased»).

2. ~~**Journal im Transkript**~~ — GEBAUT 2026-09-10/11 (Runs je
   Whisper-Lauf, Import, Editor-Sitzung; im Container als RunRecords
   mit enrichs Kette). Offen daraus: der Hinweis in der App, dass ein
   Run offen ist, und `enrich_core`-Umzug von `zotero.py` abwarten für
   die Metadaten-Schicht (enrich-Backlog 000).

2a. ~~**Neu vendoren nach enrichs Kopfzeile**~~ — ERLEDIGT 2026-09-11
   (textsatz@94ba895; Kopfzeile «Titel · Datum» aus dem Transkript).
   OFFEN daraus: **Metadaten-Panel mit Zotero** — `enrich_core.zotero`
   liegt jetzt in enrich-core (24333d4): Einwilligung als Einstellung
   (Semantik `zotero_consent`, im Datenfluss nennen), lokale
   `zotero.sqlite` immutable lesen, Kandidaten nach Titel-Nähe (kein
   PDF-Hash), Typ Interview bevorzugt, Rollenwahl je Creator wegen
   Pseudonymisierung, Snapshot als `zotero.json` (`origin: source`,
   Verknüpfung als Run `human`), im Export registriert; Kopfzeile dann
   mit Interviewer:in und Citekey über `kopfzeile_aus_meta`.

3. **Import-Dialog für Ordner-Dossiers ohne enrich.app.** Ohne die UTI
   (kommt mit dem nächsten Build über `src-tauri/Info.plist`) ist ein
   `.enrich`-Verzeichnis im Dateidialog nicht wählbar; Drag & Drop
   geht. Nach dem Build prüfen, ob der Dialog das Package anbietet;
   sonst zweiter Knopf «Ordner wählen …» (`pickOrdner` gibt es).

4. **Anleitung, Schritt Exportieren:** Satz zum `.qdpx.zip` — vor dem
   Import in ATLAS.ti entpacken; `.qdpx` und `Media`-Ordner müssen
   nebeneinander liegen (User 2026-09-10). Viersprachig, Seite und
   App-Blatt.
