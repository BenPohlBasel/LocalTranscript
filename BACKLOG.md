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
   Run offen ist.

2a. ~~**Neu vendoren nach enrichs Kopfzeile**~~ — ERLEDIGT 2026-09-11
   (textsatz@94ba895; Kopfzeile «Titel · Datum» aus dem Transkript).
   ~~OFFEN daraus: **Metadaten-Panel mit Zotero**~~ — GEBAUT 2026-09-11
   (Subtab «Metadaten» neben Suchen, Einstellungs-Karte, Schicht
   `source/zotero.json`, Kopfzeile mit Interviewer:in und Citekey;
   CHANGELOG). Zugleich beide Umgehungen gestrichen, die enrich
   angemahnt hatte (Installations-Kennung als `agent.user`, Rollen-
   Korrektur nach dem Textsatz): der Setzer bekommt die Transkript-
   Schicht (`baue_struktur_dossier(transkript=…)`), enrich packt die
   Sendung (`Dossier.pack(profile="handover")`), Namen nach Anhang A.
   Zwei neue Befunde an enrich (BACKLOG 000 dort): `producer` wird
   hart auf «enrich» gesetzt (hier nach dem Packen zurückgeschrieben),
   `by` der übernommenen Quelle kommt aus dem Setzer-Run.
   Ursprüngliche Spezifikation: `enrich_core.zotero`
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

5. **ZURÜCKGESTELLT (User-Entscheid 2026-09-11): Bibliothek als
   Arbeitsdossiers (`<name>.enrich/`, Format 2, unkomprimiert).**
   Nicht bauen, solange kein konkreter Bedarf besteht — etwa dass
   enrich und LocalTranscript DASSELBE Dossier abwechselnd bearbeiten
   sollen (dann zuerst die Sperr-Konvention). Begründung: das Ziel ist
   erreicht — was die App verlässt, ist ein Format-2-Dossier, das
   enrich liest (`Dossier.uebernehmen`, Inventar und Kette sauber). Der
   Umbau kaufte Eleganz («eine Form innen wie aussen») zum Preis, dass
   jeder Autosave durch enrich-cores `write_layer` liefe (Kopplung des
   Speicherpfads an eine Bibliothek, die sich schnell bewegt), einer
   Migration echter Studien-Daten, dreier Vorarbeiten in enrich und
   einer Bibliothek, die Forscher:innen nicht mehr als EINE
   `transkript.json` je Eintrag lesen können. Netto kein Code-Gewinn
   (`format2.py` fiele weg, `bibliothek.py` auf `Dossier` + Migration
   käme dazu). Ursprüngliche Skizze, falls es doch nötig wird — eine
   Form in beide Richtungen:
   Export = `Dossier.pack(profile="handover")`, Import =
   `Dossier.uebernehmen`; kein Konverter mehr (`format2.py` fällt
   weitgehend weg), Journal/Locks/Zotero werden direkt geschrieben,
   die Bibliothek ist im Finder ein Ordner mit Packages (UTI steht).
   Entscheide: Arbeitsdossier hält NUR `source/` (Transkript, Audio,
   Zotero) + Manifest — die Textschichten (PDF, Layout, T0/T1,
   Zeitkarte) entstehen beim Packen, nicht bei jedem Speichern (8 s
   beim 3,4-h-Workshop); Bibliotheksordner GETRENNT vom enrich-
   Projektordner (gleiche Form, anderer Ort — enrichs Index und
   Warteschlange sollen nicht auf halbfertige Transkripte losgehen;
   Übergabe bleibt ein bewusster Akt); Audio schon beim Anlegen als
   mp3 (`pack` transkodiert nicht); neue Segmente/Sprecher als
   `sg-`/`sp-`-ULID, Altbestand bleibt; `liste()` aus dem Manifest;
   Migration bestehender Bibliotheken idempotent mit Sicherung
   (`<stamp>/transkript.json` → `<stamp>.enrich/source/transcript.json`,
   Audio nach `source/`, Journal übernommen); `_papierkorb/` bleibt
   ausserhalb. **Blockiert durch zwei Punkte in enrich-core (dort
   BACKLOG 000, Eintrag «Arbeitsdossier für LocalTranscript»):**
   (1) History beschränken auf die letzten 10 Stände je Schicht
   (User-Entscheid 2026-09-11, wie `HISTORY_MAX` hier) — `write_layer`
   rettet heute bei JEDER Schreibung nach `_history/` ohne Grenze; der
   Editor sichert debounced alle paar Sekunden → hunderte MB pro Stunde
   bei 1148 Segmenten. (2) Ein
   Dossier ohne Textschichten muss für enrich ein gültiger Zustand
   sein: Job «Text setzen aus der Transkript-Schicht» (Bausteine
   `segmente_als_turns`, `textsatz(transkript=…)` sind da). Dazu:
   LocalTranscript nimmt enrichs Sidecar-Sperre (`<name>.enrich.lock`,
   O_EXCL) und respektiert sie. Aufwand hier danach ~1 Tag
   (`bibliothek.py` auf `Dossier`, Migration, Tests).

6. **Bibliothek in iCloud Drive erkennen (Befund 2026-09-11).** Auf dem
   Entwickler-Mac liegt `~/Documents` in iCloud Drive («Schreibtisch &
   Dokumente»); bei 99 % voller Platte lagert macOS Dateien aus
   (`SF_DATALESS`) — ein 1,6-GB-Modell im Bibliotheksordner brauchte 16 s
   zum Öffnen, `/api/models` hing. Modelle sind seit 2.3.0 abgefangen;
   für Audio/Transkripte gilt dasselbe Risiko, und die Aufnahmen werden
   dann nach iCloud synchronisiert (die Blätter sagen das). In den
   Einstellungen warnen, wenn der Bibliotheksordner unter iCloud liegt
   (Prüfung: Pfad unter `~/Library/Mobile Documents/` oder `~/Documents`/
   `~/Desktop` bei aktivem Desktop-&-Dokumente-Sync — `brctl`/
   `com.apple.icloud.desktop`-Marker prüfen), und einen Ort ausserhalb
   anbieten.

## Gemessen, nicht gebaut

- **MLX als zweiter Runtime (User-Frage 2026-09-11: «beschleunigt mit
  MLX wäre gut»).** Messung auf diesem Mac, 5-min-Ausschnitt einer echten
  Aufnahme, large-v3-turbo: `whisper-cli` (whisper.cpp, Metal, Flags der
  App) **15,9 s**, `mlx_whisper` 0.4.3 (mlx 0.32, GPU) **13,1 s** — gleicher
  Wortlaut (633 Wörter). MLX ist ~18 % schneller: pro Stunde Audio ~3,2 min
  statt ~2,6 min. Dafür ein zweiter Runtime-Pfad (mlx, numba, tiktoken im
  Bundle, ~200 MB), ein zweites Modellformat im Modelle-Ordner, und für
  Modelle mit eigenem Tokenizer (CrisperWhisper) eine gepatchte
  Tokenizer-Ladung, weil mlx_whisper das OpenAI-Vokabular fest verdrahtet.
  Entscheid: nicht bauen, solange whisper.cpp nicht zurückfällt. Falls
  Beschleunigung nötig wird, zuerst whisper.cpps CoreML-Encoder prüfen
  (die gebündelte `whisper-cli` ist ohne CoreML gebaut — `strings` findet
  keinen CoreML-Bezug); das ist derselbe Runtime, nur ein Build-Flag.
  Nebenbefund: `-nt` (ohne Zeitstempel) lässt whisper.cpp ganze Fenster
  fallen (518 statt 633 Wörter) — die App setzt es nicht, richtig so.
- **CrisperWhisper** (nyralabs, Basis large-v3, CC-BY-NC-4.0) läuft als
  eigenes Modell — aber nur mit `scripts/hf-nach-ggml.py` (Tokentabelle
  nach ID, s. Kopf des Skripts); der unveränderte whisper.cpp-Konverter
  liefert Kauderwelsch.
- **Schweizerdeutsch: Flurin17/whisper-large-v3-turbo-swiss-german**
  (turbo-Basis, ~301 h, CC-BY-NC-4.0, bfloat16 — darum die bf16-Hebung im
  Konverter) läuft als eigenes Modell (`ggml-large-v3-turbo-swiss-
  german.bin`, 1,6 GB). Auf dem Podcast-Ausschnitt (Standarddeutsch):
  17,3 s gegen 15,9 s turbo, 634/633 Wörter, gleicher Inhalt; Unterschiede
  nur Schweizer Schreibung («ss» statt «ß») und Guillemets «…» um jede
  Äusserung (Untertitel-Stil der Trainingsdaten — für den Editor evtl.
  beim Import strippen, wenn das Modell in Gebrauch kommt). Der echte
  Nutzen zeigt sich erst auf Mundart-Aufnahmen — noch nicht gemessen.
