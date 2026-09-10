# LocalTranscript — App-Blatt für Forschende

Was die App ist, welche KI darin was tut, wo der Code liegt, und warum
das Transkript den Rechner nicht verlässt. Zum Weitergeben an
Projektleitung, Ethikkommission oder Kolleg:innen. Stand 10. September
2026, Version 2.2.0.

## Was die App tut

LocalTranscript wandelt Audioaufnahmen — Interviews, Gruppengespräche,
Workshops — in Text mit Zeitmarken und Sprecherzuordnung um. Danach
wird das Transkript in einem Editor korrigiert, Sprecher werden
benannt, Namen ersetzt, und das Ergebnis wird für die Auswertung
exportiert (ATLAS.ti, MAXQDA, NVivo über REFI-QDA; enrich; WebVTT,
CSV, Text). Alles davon geschieht auf dem eigenen Mac.

## Welche KI hier was macht

| Baustein | Aufgabe | Herkunft, Lizenz | Läuft wo |
|---|---|---|---|
| whisper.cpp mit Modell `large-v3-turbo` | Spracherkennung: Audio → Text mit Zeitmarken | Modell von OpenAI (MIT), Laufzeit whisper.cpp (MIT) | lokal, auf der Grafikeinheit des Macs |
| silero-vad | Sprachaktivität: erkennt, wo überhaupt gesprochen wird | MIT | lokal |
| SpeechBrain ECAPA-TDNN | Sprechertrennung: berechnet Stimmprofile und gruppiert sie zu Sprechern | Apache-2.0 | lokal |

Alle drei Modelle liegen im Programmpaket. Es gibt keinen Zugang zu
einem KI-Dienst, kein Konto, keinen Schlüssel.

**Keine generative KI.** Nichts wird zusammengefasst, umformuliert oder
interpretiert. Die App liefert, was gesagt wurde — nicht, was gemeint
war.

**Grenze der Spracherkennung.** Whisper ist ein neuronales Modell. Wo
es nichts versteht (Nebengeräusche, Dialekt, Überlappungen), kann es
Wörter setzen, die nicht gesagt wurden. Ein Transkript aus
LocalTranscript ist ein **Rohtranskript**, das gegen die Aufnahme geprüft
werden muss; der Editor ist dafür gebaut. Standarddeutsch, Französisch,
Italienisch und Englisch werden gut erkannt, Schweizerdeutsch lückenhaft
— die Sprecher werden trotzdem sauber getrennt.

## Wo der Code liegt

- Quellcode: <https://github.com/BenPohlBasel/LocalTranscript>
- Lizenz: AGPL-3.0-or-later — freie Software, darf genutzt, geprüft,
  verändert und weitergegeben werden
- Entwickelt am B/IAS – Basel Institut für angewandte Stadtforschung,
  Beckenweg 6, 4056 Basel, <https://bias.city>
- Installationspaket: signiert und notarisiert mit Apple Developer ID,
  Prüfsumme bei jedem Release auf GitHub
- Wer der Binärdatei nicht traut: Das Repository enthält die
  vollständige Build-Kette, die App lässt sich selbst bauen

## Das Transkript verlässt den Rechner nicht

- Die App baut **keine ausgehenden Netzverbindungen** auf: keine
  Telemetrie, keine Nutzungsstatistik, keine Update-Prüfung, keine
  eigenen Absturzberichte.
- Ihr interner Dienst hört nur auf der Loopback-Adresse `127.0.0.1`
  des eigenen Rechners und weist jede Anfrage von anderswo ab. Der
  Code dafür steht in `backend/src/localtranscript/main.py` — für jede
  Person nachlesbar.
- Audio und Transkript liegen ausschliesslich im gewählten
  Bibliotheksordner. Gelöschtes wandert in einen Papierkorb-Ordner
  innerhalb der Bibliothek, bis er geleert wird.
- Kein Server, kein Cloud-Dienst, kein Benutzerkonto, kein
  Auftragsverarbeiter, kein Drittlandtransfer — weil nichts übermittelt
  wird.

Was damit **nicht** abgedeckt ist: Backups des Macs (Time Machine,
iCloud Drive für den Dokumente-Ordner) und die Diagnosedaten von macOS
selbst folgen den Systemeinstellungen, nicht der App. Wer den
Bibliotheksordner in einen synchronisierten Ordner legt, synchronisiert
die Aufnahmen.

## Namen anonymisieren

- **Sprecher umbenennen:** Ein Name im Sprecher-Panel gilt für alle
  Segmente dieser Person — aus «Sprecher 1» wird «B3» in einem Schritt.
- **Namen im Text:** «Suchen und Ersetzen» findet einen Namen in allen
  Segmenten, zeigt jeden Treffer im Zusammenhang und ersetzt einzeln
  oder alle auf einmal — auch, wenn das Transkript den Namen über eine
  Zeile getrennt hat.
- **Was die App nicht entscheidet:** welche Angaben zu ersetzen sind —
  Orte, Arbeitgeber, Ereignisse. Das bleibt eine Entscheidung der
  Forschenden.
- **Die Aufnahme bleibt, was sie ist.** Pseudonymisiert wird der Text.
  Exporte als REFI-QDA (`.qdpx.zip`) und enrich-Dossier (`.enrich`)
  enthalten die Audiodatei mit Stimme und Klarnamen; WebVTT, CSV und
  Text enthalten nur den Text. Wer nur pseudonymisierte Daten weitergeben
  will, gibt ein Textformat weiter.

## Für den Methodenteil

> Die Aufnahmen wurden mit LocalTranscript 2.2.0 (B/IAS Basel,
> AGPL-3.0; Spracherkennung whisper.cpp mit dem Modell large-v3-turbo,
> Sprechertrennung mit SpeechBrain ECAPA-TDNN) vollständig lokal auf
> einem Rechner der Forschungsgruppe transkribiert, ohne Übermittlung an
> externe Dienste. Die Rohtranskripte wurden anschliessend gegen die
> Aufnahme korrigiert und pseudonymisiert.

---

Quelle: <https://github.com/BenPohlBasel/LocalTranscript> (Ordner
`site/docs`). Das Blatt darf frei verwendet und angepasst werden.
