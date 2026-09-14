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

7. ~~**Zotero-Rollen nicht übersetzen (User-Entscheid 2026-09-11).**~~ GEBAUT 2026-09-11 (CHANGELOG 2.4.0). Die
   Rollen kommen aus Zotero (creatorType: interviewer, interviewee,
   director, producer, castMember, podcaster, guest …) und sollen im
   Metadaten-Reiter und in der verknüpften Ansicht so stehen, wie Zotero
   sie führt — **englisch, für alle Typen gleich**, keine Übersetzung
   auch nicht der sechs Interview-Rollen. Umbau: `rolleName()` in
   EditorModule.tsx gibt den Schlüssel zurück, die i18n-Keys
   `ed.meta.rolle.*` fallen weg; Vorauswahl bleibt eine Liste bekannter
   Rollen (interviewee, castMember, guest … nicht vorausgewählt).
   Optional die Hilfezeile «Rollen wie in Zotero» viersprachig.

8. ~~**Video (User-Konzept 2026-09-11).**~~ GEBAUT 2026-09-11 (CHANGELOG 2.4.0; geprüft in Chromium und WebKit: Bild folgt dem Ton auf < 0,25 s, Sprung, Einfrieren ab 2×). Konzept: Eingang: **keine Grenze bei
   Dauer oder Auflösung, nur Dateigrösse ≤ 10 GB.** Es wird **nichts
   umgewandelt**: angenommen wird, was der WKWebKit nativ spielt — H.264
   und HEVC in MP4/MOV/M4V (AV1 nur, wo die Hardware es dekodiert);
   alles Ältere oder Fremde (VP8/VP9/webm, mkv, AVI/DivX/WMV, MPEG-2,
   …) wird **abgewiesen mit Hinweis** («Video wird nicht umgewandelt —
   extern als H.264/MP4 exportieren, z. B. HandBrake»). ffprobe liefert
   Codec, Container und Grösse für die Prüfung. Ton wie heute nach mp3,
   Bild als eigene Datei unverändert in den Eintrag kopiert.
   **Bild und Ton getrennt: Ton führt, Bild folgt** — der Audioplayer
   bleibt Zeitquelle, das Video läuft stumm und wird alle ~250 ms auf
   `audio.currentTime` nachgezogen, `playbackRate` 0,5–2× mit; bei
   Shuttle (J/K/L ≥ 2×, rückwärts, Sprünge) Bild einfrieren und
   weichzeichnen, danach scharf und weiter (WebKit kann nicht rückwärts;
   Suchen landet auf dem vorigen Keyframe — bei langen Keyframe-
   Abständen einer Aufnahme dauert der Sprung entsprechend, das ist der
   Preis dafür, nichts umzuwandeln). **Ort des Bilds (User 2026-09-11):
   in der rechten Seitenleiste unten, unter den Sprechern — fest
   eingebaut, nicht schwebend, ohne Knöpfe.** Ein stummes `<video>` im
   Sprecher-Panel, Breite = Panelbreite (resizable wie das Panel),
   keine Controls; einzige Bedienung ist der Audioplayer. Kein PiP, kein
   zweites Fenster, kein Icon in der Fusszeile (die Idee ist damit vom
   Tisch — falls doch je gewünscht, wäre PiP der Weg). Im Reiter
   «Suchen»/«Metadaten» bleibt das Bild aus dem Sichtfeld, Ton läuft
   weiter. Export: qdpx-Häkchen «Video mitgeben»
   (`VideoSource`, Datei im `Media/`-Ordner); **enrich bleibt Ton
   (mp3) — Entscheid User 2026-09-11: «enrich hat nur Ton».**
   Datenschutzblätter ×4 um Video (Gesichter, biometrisch) ergänzen;
   iCloud-Warnung (6) wird damit dringend. **Bestätigt 2026-09-11: ohne Umwandlung, auch nicht optional** —
   Original bleibt Original; Sprünge kosten bei langen Keyframe-
   Abständen bis ~0,5 s, das Einfrieren beim Shuttle fängt es ab.
   Einzige denkbare Ausnahme, falls die Probe sie nötig macht: ein
   verlustfreies Umpacken des Containers (mov → mp4, Sekunden, keine
   Bildänderung). Aufwand ~2 Tage; erster Schritt eine halbe Stunde
   Probe: Keyframe-Abstände echter Aufnahmen und Sync-Jitter Ton→Bild
   im WKWebKit messen.

9. ~~**Notarisierung aus Tauri herausnehmen (Befund Release 2.4.1,~~ GEBAUT 2026-09-12 (scripts/notarize-app.mjs, build-dmg.mjs, npm run release). Ursprünglich:
   2026-09-11 abends).** Apples Warteschlange brauchte 22 min statt
   12–15; Tauris interne Wartezeit lief ab und `tauri build` brach mit
   leerer Meldung ab — die Einreichung lief bei Apple weiter, das DMG
   entstand nicht. Nachlauf von Hand: `notarytool wait <id>`, `stapler
   staple` auf die unveränderte .app, DMG per `hdiutil` (UDZO,
   Applications-Link), `codesign` des DMG, `notarize-dmg.mjs`. Umbau:
   `tauri build` OHNE APPLE_*-Variablen (nur signieren), danach eigenes
   Skript `scripts/notarize-app.mjs` mit `--timeout 2h`, Stapeln, dann
   DMG wie gehabt; `npm run release` entsprechend verketten. Vorteil:
   ein Ticket-Timeout kostet nie mehr den Build. Das hdiutil-DMG ist
   2,1 GB statt Tauris 1,9 GB (Kompressionsstufe) — akzeptabel, oder
   Tauris `bundle_dmg.sh` aus dem target-Ordner nachnutzen.

10. ~~**Sprecherzahl je Batch-Eintrag, Pflichtangabe**~~ — GEBAUT
   2026-09-13 (CHANGELOG 2.5.0): Dateien sammeln sich in einer
   Warteschlange, jede mit eigener Auswahl; der Startknopf bleibt
   gesperrt, solange eine Angabe fehlt, und der gewählte Wert steht in
   `quelle.sprecherzahl` und im Journal (`by.speakers`). Die globale
   Auswahl ist nur noch Vorgabe für neu hinzugefügte Dateien; beim Ablegen
   mehrerer Dateien bleibt die Wahl offen.

11. ~~**Stimmenvorschau bricht die vorherige ab**~~ — GEBAUT 2026-09-13
   (CHANGELOG 2.5.0): eine Referenz auf den laufenden Klang, jeder neue
   Klick hält den alten an, ein zweiter Klick auf dieselbe Stimme stoppt,
   und beim Verlassen des Panels ist Ruhe.

12. **Lizenzzeile in der App nachziehen (User 2026-09-14).** README,
   Website und die vier Blätter nennen seit dem 14. September beide
   Modelle — pyannote community-1 und WeSpeaker ResNet34 — unter CC BY
   4.0 samt dem Hinweis, dass Argmax sie nach Core ML umgewandelt und
   quantisiert hat. CC BY 4.0 verlangt neben der Namensnennung
   ausdrücklich die Kennzeichnung von Änderungen. `st.lizenzen.text` in
   `frontend/src/lib/i18n.ts` trägt noch die kürzere Fassung; sie ist
   nicht falsch, nur unvollständig. Beim nächsten Release in allen vier
   Sprachen nachziehen — der Wortlaut steht im CHANGELOG unter
   «Unreleased» und in Punkt 10 der Recherche.

13. **App umbenennen; Repository ist umgezogen (User 2026-09-14).** Der
   Name «LocalTranscript» kollidiert mit einer geschlossenen Meeting-App
   aus demselben Umfeld (github.com/localtranscript, intern «Secure
   Transcript»); belegbar war unser Name 36 Stunden früher öffentlich
   (GH Archive: unser erster Push 25.4.2026 21:58 UTC, sein erstes
   Ereignis 27.4.2026 10:26 UTC). Das Repository liegt seit dem
   14. September unter github.com/bias-city/LocalTranscript; die alte
   Adresse leitet weiter. Website, Blätter und LICENSE-docs zeigen schon
   auf den neuen Ort. Bei der Umbenennung mit erledigen: die Verweise in
   `frontend/src/App.tsx` und `EinstellungenModule.tsx`, Produktname,
   Bundle-Kennung, Bibliotheksordner (mit Übernahme bestehender
   Installationen), Website-Adresse bias.city/localtranscript, Texte in
   vier Sprachen. Vorher den neuen Namen gegen GitHub, Domains, App-Namen
   und die Markenregister Swissreg und EUIPO prüfen.

## Gemessen, nicht gebaut

- **Diarisierung: welches Modell? (Recherche 2026-09-12/13)** — GEBAUT
  2026-09-13 mit Argmax SpeakerKit, siehe CHANGELOG 2.5.0. Verworfen:
  **precision-2** (Nutzungsbedingungen verbieten Einbetten und
  Weitergabe, dazu AGPL-Konflikt), **pyannote.audio 4.0 in Python**
  (Telemetrie müsste hart abgeschaltet und mit einem Test gesichert
  werden, torch bliebe im Bundle), **DiariZen/Reverb** (nicht-
  kommerzielle Lizenz), **sherpa-onnx** (ältere segmentation-3.0),
  **NVIDIA Sortformer** (höchstens vier Sprecher). Daraus war die
  **Lizenz der Gewichte** offen — GEKLÄRT am 13. September 2026: das
  Repository `argmaxinc/speakerkit-coreml` trägt jetzt `license:
  cc-by-4.0` im Frontmatter (Commit `556fc52`, 20:20 UTC; die Revision
  davor, `36b58e2`, sagte noch `mit`), und die Modellkarte sagt im Text:
  «SpeakerKit the Swift framework has MIT license. The models SpeakerKit
  is built on have CC BY 4 license.» Die frühere Proprietary-Notiz war
  schon am 7. Mai 2026 entfernt worden (`86ec9c9`). CC BY 4.0 erlaubt
  kommerzielle Nutzung und Weitergabe ohne ShareAlike; Pflicht ist die
  Namensnennung UND die Kennzeichnung von Änderungen — Argmax hat die
  Gewichte nach Core ML umgewandelt und quantisiert. Genau so steht es
  jetzt in den Lizenzen der App, im README, auf der Website und in den
  vier Blättern. Belege:
  https://huggingface.co/argmaxinc/speakerkit-coreml/blob/556fc52a13327837688f02289457cded017802e9/README.md

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
