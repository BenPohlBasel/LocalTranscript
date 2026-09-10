# LocalTranscript — Beschreibung der Datenverarbeitung

Textbaustein zum Einfügen in ein Verfahrensverzeichnis, eine
Datenschutz-Folgenabschätzung, einen Ethikantrag oder einen
Datenmanagementplan. Stand 10. September 2026, LocalTranscript 2.2.0.
Angaben in `[eckigen Klammern]` ergänzt die verantwortliche Stelle.

Der Text beschreibt, was die Software tut und was sie nicht tut. Die
rechtliche Einordnung der eigenen Verarbeitung — nach DSGVO oder
revidiertem Schweizer DSG — nimmt die verantwortliche Stelle vor; der
Text ersetzt keine Rechtsberatung.

---

## 1. Eingesetzte Software

LocalTranscript, Version `[2.2.0]`. Freie Software unter
AGPL-3.0-or-later, entwickelt am B/IAS – Basel Institut für angewandte
Stadtforschung. Quellcode öffentlich unter
<https://github.com/BenPohlBasel/LocalTranscript>. Die Software läuft
als lokale Anwendung auf macOS (Apple Silicon) und wird von der
verantwortlichen Stelle selbst installiert und betrieben.

## 2. Zweck der Verarbeitung

Umwandlung von Audioaufnahmen `[z. B. leitfadengestützte Interviews im
Projekt …]` in Text mit Zeitmarken und Sprecherzuordnung, zur
anschliessenden qualitativen Auswertung `[in ATLAS.ti / MAXQDA / NVivo /
enrich / …]`.

## 3. Betroffene Personen und Datenkategorien

Betroffen sind die aufgenommenen Personen `[Interviewpartner:innen,
Teilnehmende an Gruppengesprächen, …]`. Verarbeitet werden
Sprachaufnahmen (Stimme und Gesprächsinhalt) sowie die daraus erzeugten
Transkripte mit Zeitmarken und Sprecherzuordnung. Je nach
Gesprächsinhalt können besondere Kategorien personenbezogener Daten
betroffen sein `[ja / nein: …]`.

## 4. Datenfluss einer Transkription

1. **Eingabe.** Die Audiodatei wird vom lokalen Dateisystem des
   Endgeräts eingelesen (MP3, WAV, M4A, OGG, FLAC).
2. **Verarbeitung.** Spracherkennung (whisper.cpp, Modell
   large-v3-turbo) und Sprechertrennung (silero-vad, SpeechBrain ECAPA)
   laufen im Prozess der Anwendung auf dem Prozessor bzw. der Grafikkarte
   des Endgeräts. Sämtliche Modelle sind im Programmpaket enthalten; beim
   ersten Start wird nichts nachgeladen.
3. **Ablage.** Je Transkript entsteht ein Ordner am gewählten
   Speicherort `[Pfad, z. B. ~/Documents/LocalTranscript]` mit einer
   Kopie des Audios, der kanonischen Transkriptdatei (JSON),
   Verlaufsschnappschüssen bei jedem Speichern und den abgeleiteten
   Exporten. Temporäre Arbeitsdateien werden nach jedem Lauf entfernt.
4. **Netz.** Die Software baut keine ausgehenden Netzverbindungen auf:
   keine Telemetrie, keine Nutzungsstatistik, keine Update-Prüfung,
   keine eigenen Absturzberichte. Der interne Dienst der Anwendung
   bindet ausschliesslich an die Loopback-Adresse `127.0.0.1` und weist
   Anfragen anderer Hosts ab (HTTP 421). Diagnosedaten des
   Betriebssystems macOS unterliegen dessen Systemeinstellungen, nicht
   der Software.
5. **Ausgabe.** Exportdateien (WebVTT, CSV, Text, REFI-QDA `.qdpx.zip`,
   enrich-Dossier `.enrich`) werden dorthin geschrieben, wo die
   bedienende Person sie speichert. **REFI-QDA- und enrich-Exporte
   enthalten die Audioaufnahme.** Ihre Weitergabe ist eine Weitergabe
   der Aufnahme.

## 5. Ort der Verarbeitung

Ausschliesslich auf dem Endgerät `[Gerät, Standort]` in der Sitzung der
angemeldeten Person. Es gibt keinen Server, keinen Cloud-Dienst und kein
Benutzerkonto.

## 6. Empfänger, Auftragsverarbeitung, Drittlandübermittlung

Keine. Da keine Daten übermittelt werden, gibt es weder Empfänger noch
Auftragsverarbeiter noch eine Übermittlung in ein Drittland. Eine
Weitergabe findet nur statt, wenn die verantwortliche Stelle
Exportdateien selbst weitergibt `[an …, auf dem Weg …]`.

## 7. Speicherdauer und Löschung

Aufbewahrung der Aufnahmen und Transkripte: `[Frist, Grundlage]`.
Löschen in der Anwendung verschiebt einen Eintrag in einen
Papierkorb-Ordner innerhalb der Bibliothek (`_papierkorb`); endgültig
entfernt wird er erst durch Leeren dieses Ordners `[durch wen, wann]`.
Verlaufsschnappschüsse liegen im Ordner des jeweiligen Transkripts und
werden mit ihm gelöscht. Sicherungskopien des Endgeräts `[Time Machine,
…]` unterliegen der Löschregel der Stelle.

## 8. Technische und organisatorische Massnahmen

Von der verantwortlichen Stelle zu erbringen, da die Software selbst
keine Zugriffssteuerung mitbringt:

- Verschlüsselung des Datenträgers, z. B. FileVault: `[aktiv seit …]`
- Zugriffsschutz des Endgeräts (Anmeldung, Bildschirmsperre): `[…]`
- Pseudonymisierung vor jeder Weitergabe — im Editor der Anwendung
  lassen sich Sprecher umbenennen und Namen im Text per Suchen und
  Ersetzen tauschen; die Entscheidung, was zu ersetzen ist, trifft die
  bearbeitende Person: `[Verfahren, Zuständigkeit]`
- Regel für Sicherungskopien: `[…]`
- Regel für die Weitergabe von Exportdateien, insbesondere solcher mit
  Audio: `[…]`

## 9. Rechtsgrundlage und Information der Betroffenen

`[Einwilligung / berechtigtes Interesse / Forschungsprivileg nach …;
Informationsschreiben vom …]`. Die Software trägt hierzu nichts bei.

## 10. Nachprüfbarkeit

Die Aussagen in Abschnitt 4 lassen sich am Quellcode prüfen: Die
Bindung des internen Dienstes an `127.0.0.1` und die Abweisung fremder
Hosts stehen in `backend/src/localtranscript/main.py`. Das Repository
enthält die vollständige Build-Kette bis zum signierten Installations-
paket; wer der ausgelieferten Binärdatei nicht traut, kann sie selbst
erzeugen.

---

Quelle dieses Textes: <https://github.com/BenPohlBasel/LocalTranscript>
(Ordner `site/docs`). Er darf frei verwendet und angepasst werden.
