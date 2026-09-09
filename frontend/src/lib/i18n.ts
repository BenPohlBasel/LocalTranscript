// UI-Sprache (enrich-Mechanik): EINE Wörterbuch-Quelle, Umschaltung ohne
// Reload via useSyncExternalStore; fehlende Schlüssel fallen LAUT auf
// den Schlüsselnamen zurück. de/en/fr/it (fr/it-Lücke → en, nie still
// deutsch).
import { useSyncExternalStore } from "react";
import { KEYS, lget, lset } from "./storage";

export type Sprache = "de" | "en" | "fr" | "it";

const _GUELTIG = new Set(["de", "en", "fr", "it"]);
let aktuelle: Sprache = (_GUELTIG.has(lget(KEYS.sprache) ?? "")
  ? (lget(KEYS.sprache) as Sprache) : "de");
const hoerer = new Set<() => void>();

export function setSprache(s: Sprache): void {
  aktuelle = s;
  lset(KEYS.sprache, s);
  hoerer.forEach((h) => h());
}
export function getSprache(): Sprache { return aktuelle; }
export function useSprache(): Sprache {
  return useSyncExternalStore(
    (cb) => { hoerer.add(cb); return () => hoerer.delete(cb); },
    () => aktuelle);
}

type Eintrag = { de: string; en: string; fr?: string; it?: string };

const W: Record<string, Eintrag> = {
  "app.titel": { de: "LocalTranscript", en: "LocalTranscript",
    fr: "LocalTranscript", it: "LocalTranscript" },
  "app.untertitel": { de: "Offline-Transkription mit Sprechererkennung",
    en: "Offline transcription with speaker detection",
    fr: "Transcription hors ligne avec détection des locuteurs",
    it: "Trascrizione offline con riconoscimento dei parlanti" },
  "app.boot": { de: "Backend startet …", en: "Backend starting …",
    fr: "Démarrage du backend …", it: "Avvio del backend …" },
  "app.bootfehler": { de: "Backend nicht erreichbar",
    en: "Backend unreachable", fr: "Backend inaccessible",
    it: "Backend non raggiungibile" },
  "app.nochmal": { de: "Erneut versuchen", en: "Try again",
    fr: "Réessayer", it: "Riprova" },
  "tab.ai": { de: "AI-Transkript", en: "AI Transcript",
    fr: "Transcription IA", it: "Trascrizione IA" },
  "tab.editor": { de: "Human-Editor", en: "Human Editor",
    fr: "Éditeur humain", it: "Editor umano" },
  "ai.batch": { de: "Batch-Liste", en: "Batch list",
    fr: "Liste des lots", it: "Elenco batch" },
  "he.leer": {
    de: "Noch keine Transkripte — im AI-Transkript-Tab transkribieren oder hier importieren.",
    en: "No transcripts yet — transcribe in the AI Transcript tab or import here.",
    fr: "Pas encore de transcriptions — transcrivez dans l'onglet Transcription IA ou importez ici.",
    it: "Ancora nessuna trascrizione — trascrivi nella scheda Trascrizione IA o importa qui." },
  "he.audiowahl": {
    de: "Audio dazu wählen (mp3 — Abbrechen, wenn keins)",
    en: "Choose matching audio (mp3 — cancel if none)",
    fr: "Choisir l'audio associé (mp3 — annuler si aucun)",
    it: "Scegli l'audio associato (mp3 — annulla se nessuno)" },
  "allg.bearbeiten": { de: "Bearbeiten", en: "Edit", fr: "Modifier",
    it: "Modifica" },
  "tab.einstellungen": { de: "Einstellungen", en: "Settings",
    fr: "Réglages", it: "Impostazioni" },

  // First-Run
  "firstrun.titel": { de: "Speicherort wählen", en: "Choose storage",
    fr: "Choisir l'emplacement", it: "Scegli la posizione" },
  "firstrun.text": {
    de: "Wo sollen deine Transkripte liegen? Je Transkript entsteht ein Ordner mit Audio, Text und Verlauf.",
    en: "Where should your transcripts live? Each transcript gets a folder with audio, text and history.",
    fr: "Où placer vos transcriptions ? Chaque transcription reçoit un dossier avec audio, texte et historique.",
    it: "Dove salvare le trascrizioni? Ogni trascrizione ha una cartella con audio, testo e cronologia." },
  "firstrun.standard": { de: "Empfohlenen Ort verwenden",
    en: "Use recommended location", fr: "Utiliser l'emplacement recommandé",
    it: "Usa la posizione consigliata" },
  "firstrun.waehlen": { de: "Anderen Ordner wählen …",
    en: "Choose another folder …", fr: "Choisir un autre dossier …",
    it: "Scegli un'altra cartella …" },

  // Bibliothek
  "bib.drop": { de: "Audio-Dateien hierher ziehen",
    en: "Drop audio files here", fr: "Déposez les fichiers audio ici",
    it: "Trascina qui i file audio" },
  "bib.dropsub": { de: "oder klicken zum Auswählen — MP3, WAV, M4A, OGG, FLAC",
    en: "or click to choose — MP3, WAV, M4A, OGG, FLAC",
    fr: "ou cliquez pour choisir — MP3, WAV, M4A, OGG, FLAC",
    it: "o fai clic per scegliere — MP3, WAV, M4A, OGG, FLAC" },
  "bib.import": { de: "Transkript importieren (VTT/CSV) …",
    en: "Import transcript (VTT/CSV) …",
    fr: "Importer une transcription (VTT/CSV) …",
    it: "Importa trascrizione (VTT/CSV) …" },
  "bib.optionen": { de: "Optionen", en: "Options", fr: "Options",
    it: "Opzioni" },
  "bib.modell": { de: "Whisper-Modell", en: "Whisper model",
    fr: "Modèle Whisper", it: "Modello Whisper" },
  "bib.sprache": { de: "Sprache", en: "Language", fr: "Langue",
    it: "Lingua" },
  "bib.diarize": { de: "Sprechererkennung", en: "Speaker detection",
    fr: "Détection des locuteurs", it: "Riconoscimento parlanti" },
  "bib.sprecherzahl": { de: "Sprecher", en: "Speakers",
    fr: "Locuteurs", it: "Parlanti" },
  "bib.auto": { de: "Automatisch", en: "Automatic", fr: "Automatique",
    it: "Automatico" },
  "bib.trennung": { de: "Trennung", en: "Separation", fr: "Séparation",
    it: "Separazione" },
  "bib.trennung.locker": { de: "Locker", en: "Loose", fr: "Souple",
    it: "Ampia" },
  "bib.trennung.normal": { de: "Normal", en: "Normal", fr: "Normale",
    it: "Normale" },
  "bib.trennung.streng": { de: "Streng", en: "Strict", fr: "Stricte",
    it: "Rigida" },
  "bib.trennung.sehr": { de: "Sehr streng", en: "Very strict",
    fr: "Très stricte", it: "Molto rigida" },
  "bib.leer": { de: "Noch keine Transkripte — Audio hierher ziehen.",
    en: "No transcripts yet — drop audio here.",
    fr: "Pas encore de transcriptions — déposez un audio ici.",
    it: "Ancora nessuna trascrizione — trascina qui un audio." },
  "bib.dauer": { de: "Dauer", en: "Duration", fr: "Durée",
    it: "Durata" },
  "bib.sprecher.n": { de: "{n} Sprecher", en: "{n} speakers",
    fr: "{n} locuteurs", it: "{n} parlanti" },
  "bib.segmente.n": { de: "{n} Segmente", en: "{n} segments",
    fr: "{n} segments", it: "{n} segmenti" },
  "bib.umbenennen": { de: "Umbenennen …", en: "Rename …",
    fr: "Renommer …", it: "Rinomina …" },
  "bib.loeschen": { de: "Löschen …", en: "Delete …", fr: "Supprimer …",
    it: "Elimina …" },
  "bib.loeschen.text": {
    de: "„{name}“ in den Papierkorb der Bibliothek verschieben?",
    en: "Move “{name}” to the library trash?",
    fr: "Déplacer « {name} » vers la corbeille ?",
    it: "Spostare “{name}” nel cestino?" },
  "bib.ordner": { de: "Im Finder zeigen", en: "Show in Finder",
    fr: "Afficher dans le Finder", it: "Mostra nel Finder" },
  "bib.jobfehler": { de: "Fehlgeschlagen: {e}", en: "Failed: {e}",
    fr: "Échec : {e}", it: "Non riuscito: {e}" },
  "bib.abbrechen": { de: "Abbrechen", en: "Cancel", fr: "Annuler",
    it: "Annulla" },
  "job.konvertiere": { de: "Konvertiere Audio …", en: "Converting audio …",
    fr: "Conversion audio …", it: "Conversione audio …" },
  "job.sprecher": { de: "Erkenne Sprecher …", en: "Detecting speakers …",
    fr: "Détection des locuteurs …", it: "Riconoscimento parlanti …" },
  "job.transkribiere": { de: "Transkribiere …", en: "Transcribing …",
    fr: "Transcription …", it: "Trascrizione …" },
  "job.transkribiere.n": { de: "Transkribiere Block {a}/{b} …",
    en: "Transcribing block {a}/{b} …",
    fr: "Transcription du bloc {a}/{b} …",
    it: "Trascrizione blocco {a}/{b} …" },
  "job.speichere": { de: "Speichere …", en: "Saving …",
    fr: "Enregistrement …", it: "Salvataggio …" },
  "job.fertig": { de: "Fertig", en: "Done", fr: "Terminé", it: "Fatto" },
  "job.abgebrochen": { de: "Abgebrochen", en: "Cancelled", fr: "Annulé",
    it: "Annullato" },
  "job.fehler": { de: "Fehler", en: "Error", fr: "Erreur",
    it: "Errore" },
  "ed.play": { de: "Abspielen (K, Leertaste, beim Tippen ⌥K)",
    en: "Play (K, Space, while typing ⌥K)",
    fr: "Lecture (K, Espace, en tapant ⌥K)",
    it: "Riproduci (K, Spazio, digitando ⌥K)" },
  "ed.pause": { de: "Pause (K, Leertaste, beim Tippen ⌥K)",
    en: "Pause (K, Space, while typing ⌥K)",
    fr: "Pause (K, Espace, en tapant ⌥K)",
    it: "Pausa (K, Spazio, digitando ⌥K)" },
  "ed.loop": { de: "Segment wiederholen (Ctrl+L)",
    en: "Loop segment (Ctrl+L)", fr: "Boucler le segment (Ctrl+L)",
    it: "Ripeti segmento (Ctrl+L)" },
  "ed.speed": { de: "Geschwindigkeit (Ctrl+X)", en: "Speed (Ctrl+X)",
    fr: "Vitesse (Ctrl+X)", it: "Velocità (Ctrl+X)" },
  "ed.rueck5": { de: "5 s zurück (J oder ←, beim Tippen ⌥J)",
    en: "5 s back (J or ←, while typing ⌥J)",
    fr: "5 s en arrière (J ou ←, en tapant ⌥J)",
    it: "5 s indietro (J o ←, digitando ⌥J)" },
  "ed.vor5": { de: "5 s vor (L oder →, beim Tippen ⌥L)",
    en: "5 s forward (L or →, while typing ⌥L)",
    fr: "5 s en avant (L ou →, en tapant ⌥L)",
    it: "5 s avanti (L o →, digitando ⌥L)" },
  "job.warte": { de: "Wartet …", en: "Waiting …", fr: "En attente …",
    it: "In attesa …" },

  // Editor
  "ed.zurueck": { de: "Bibliothek", en: "Library", fr: "Bibliothèque",
    it: "Biblioteca" },
  "ed.gespeichert": { de: "Gespeichert {t}", en: "Saved {t}",
    fr: "Enregistré {t}", it: "Salvato {t}" },
  "ed.speichert": { de: "Speichert …", en: "Saving …",
    fr: "Enregistrement …", it: "Salvataggio …" },
  "ed.speicherfehler": { de: "Speichern fehlgeschlagen: {e}",
    en: "Save failed: {e}", fr: "Échec de l'enregistrement : {e}",
    it: "Salvataggio non riuscito: {e}" },
  "ed.export": { de: "Export", en: "Export", fr: "Export",
    it: "Esporta" },
  "ed.export.qdpx": { de: "REFI-QDA für ATLAS.ti (.qdpx.zip)",
    en: "REFI-QDA for ATLAS.ti (.qdpx.zip)",
    fr: "REFI-QDA pour ATLAS.ti (.qdpx.zip)",
    it: "REFI-QDA per ATLAS.ti (.qdpx.zip)" },
  "ed.export.enrich": { de: "enrich-Dossier (.enrich.zip)",
    en: "enrich dossier (.enrich.zip)",
    fr: "Dossier enrich (.enrich.zip)",
    it: "Dossier enrich (.enrich.zip)" },
  "ed.exportiert": { de: "Exportiert: {p}", en: "Exported: {p}",
    fr: "Exporté : {p}", it: "Esportato: {p}" },
  "ed.exportfehler": { de: "Export fehlgeschlagen: {e}",
    en: "Export failed: {e}", fr: "Échec de l'export : {e}",
    it: "Esportazione non riuscita: {e}" },
  "ed.sprecher": { de: "Sprecher", en: "Speakers", fr: "Locuteurs",
    it: "Parlanti" },
  "ed.sprecher.neu": { de: "Neuer Sprecher", en: "New speaker",
    fr: "Nouveau locuteur", it: "Nuovo parlante" },
  "ed.sprecher.ohne": { de: "ohne Sprecher", en: "no speaker",
    fr: "sans locuteur", it: "senza parlante" },
  "ed.sprecher.probe": { de: "Hörprobe", en: "Sample", fr: "Extrait",
    it: "Campione" },
  "ed.sprecher.merge": { de: "Zusammenführen in …", en: "Merge into …",
    fr: "Fusionner dans …", it: "Unisci in …" },
  "ed.sprecher.leere": { de: "Allen ohne Sprecher zuweisen",
    en: "Assign to all without speaker",
    fr: "Attribuer à tous sans locuteur",
    it: "Assegna a tutti senza parlante" },
  "ed.sprecher.n": { de: "{n} Segmente", en: "{n} segments",
    fr: "{n} segments", it: "{n} segmenti" },
  "ed.tab.suchen": { de: "Suchen", en: "Find", fr: "Rechercher",
    it: "Cerca" },
  "ed.suche.was": { de: "Suchen nach", en: "Find", fr: "Rechercher",
    it: "Cerca" },
  "ed.suche.womit": { de: "Ersetzen durch", en: "Replace with",
    fr: "Remplacer par", it: "Sostituisci con" },
  "ed.suche.literal": {
    de: "Buchstabengetreu — keine Übersetzung, keine Stammformen.",
    en: "Literal — no translation, no stemming.",
    fr: "À la lettre — sans traduction ni lemmatisation.",
    it: "Alla lettera — nessuna traduzione né lemmatizzazione." },
  "ed.suche.weich": { de: "Worttrennung überlesen (Werk- statt)",
    en: "Ignore hyphenation (Werk- statt)",
    fr: "Ignorer la césure (Werk- statt)",
    it: "Ignora la sillabazione (Werk- statt)" },
  "ed.suche.gross": { de: "Groß-/Kleinschreibung beachten",
    en: "Match case", fr: "Respecter la casse",
    it: "Distingui maiuscole" },
  "ed.suche.stand": { de: "{i} von {n}", en: "{i} of {n}",
    fr: "{i} sur {n}", it: "{i} di {n}" },
  "ed.suche.keine": { de: "Keine Treffer", en: "No matches",
    fr: "Aucun résultat", it: "Nessun risultato" },
  "ed.suche.ersetzen": { de: "Ersetzen", en: "Replace",
    fr: "Remplacer", it: "Sostituisci" },
  "ed.suche.zurueck": { de: "Voriger Treffer", en: "Previous match",
    fr: "Résultat précédent", it: "Risultato precedente" },
  "ed.suche.weiter": { de: "Nächster Treffer", en: "Next match",
    fr: "Résultat suivant", it: "Risultato successivo" },
  "ed.suche.skip": { de: "Überspringen", en: "Skip", fr: "Ignorer",
    it: "Salta" },
  "ed.suche.alle": { de: "Alle ersetzen", en: "Replace all",
    fr: "Tout remplacer", it: "Sostituisci tutto" },
  "ed.suche.ersetzt": { de: "{n} ersetzt", en: "{n} replaced",
    fr: "{n} remplacés", it: "{n} sostituiti" },
  "ed.teilen": { de: "Am Cursor teilen", en: "Split at cursor",
    fr: "Scinder au curseur", it: "Dividi al cursore" },
  "ed.verbinden": { de: "Mit vorigem verbinden", en: "Merge with previous",
    fr: "Fusionner avec le précédent", it: "Unisci al precedente" },
  "ed.zeile.loeschen": { de: "Segment löschen", en: "Delete segment",
    fr: "Supprimer le segment", it: "Elimina segmento" },
  "ed.folgen": { de: "Folgen", en: "Follow", fr: "Suivre", it: "Segui" },
  "ed.abhier": { de: "Ab hier", en: "From here", fr: "À partir d'ici",
    it: "Da qui" },
  "ed.keinaudio": { de: "Kein Audio verknüpft", en: "No audio attached",
    fr: "Aucun audio associé", it: "Nessun audio collegato" },
  "ed.leer": { de: "Keine Segmente.", en: "No segments.",
    fr: "Aucun segment.", it: "Nessun segmento." },

  "st.formate": { de: "Formate", en: "Formats", fr: "Formats",
    it: "Formati" },
  "st.formate.sub": {
    de: "Austauschformate und ihre Spezifikations-Lizenzen",
    en: "Interchange formats and their specification licenses",
    fr: "Formats d'échange et licences de leurs spécifications",
    it: "Formati di scambio e licenze delle specifiche" },
  // Wortlaut bewusst zurückhaltend: „unterstützt" — es gibt KEINE
  // Zertifizierung für REFI-QDA, und Markenrechte deckt die
  // MIT-Lizenz der Spezifikation nicht ab (User 2026-09-09).
  "st.formate.refi": {
    de: "REFI-QDA (.qdpx) — LocalTranscript unterstützt den Export nach REFI-QDA; die Spezifikation steht unter der MIT-Lizenz, Copyright 2019 REFI-QDA. Keine offizielle Zertifizierung, keine Marken-Lizenz.",
    en: "REFI-QDA (.qdpx) — LocalTranscript supports export to REFI-QDA; the specification is MIT-licensed, Copyright 2019 REFI-QDA. No official certification, no trademark license.",
    fr: "REFI-QDA (.qdpx) — LocalTranscript prend en charge l'export vers REFI-QDA ; la spécification est sous licence MIT, Copyright 2019 REFI-QDA. Aucune certification officielle, aucune licence de marque.",
    it: "REFI-QDA (.qdpx) — LocalTranscript supporta l'export in REFI-QDA; la specifica è sotto licenza MIT, Copyright 2019 REFI-QDA. Nessuna certificazione ufficiale, nessuna licenza di marchio." },
  "st.formate.enrich": {
    de: "enrich-Dossier (.enrich.zip) — Format-Spezifikation unter MIT-Lizenz, BIAS.City. WebVTT (W3C) · CSV · TXT sind offen und unbeschränkt.",
    en: "enrich dossier (.enrich.zip) — format specification MIT-licensed, BIAS.City. WebVTT (W3C) · CSV · TXT are open and unrestricted.",
    fr: "Dossier enrich (.enrich.zip) — spécification du format sous licence MIT, BIAS.City. WebVTT (W3C) · CSV · TXT sont ouverts et sans restriction.",
    it: "Dossier enrich (.enrich.zip) — specifica del formato sotto licenza MIT, BIAS.City. WebVTT (W3C) · CSV · TXT sono aperti e senza restrizioni." },
  "st.formate.xsd": {
    de: "Die REFI-Schemas (XSD) liegen NICHT im Bundle — LocalTranscript schreibt nach der Spezifikation und verweist nur auf die Schema-Adresse. Damit greift die MIT-Beilagepflicht nicht.",
    en: "The REFI schemas (XSD) are NOT bundled — LocalTranscript writes to the specification and only references the schema URL. The MIT attribution requirement therefore does not apply.",
    fr: "Les schémas REFI (XSD) ne sont PAS embarqués — LocalTranscript écrit selon la spécification et ne référence que l'adresse du schéma. L'obligation d'attribution MIT ne s'applique donc pas.",
    it: "Gli schemi REFI (XSD) NON sono inclusi — LocalTranscript scrive secondo la specifica e cita solo l'indirizzo dello schema. L'obbligo di attribuzione MIT non si applica." },
  "st.link.refi": { de: "REFI-QDA-Standard", en: "REFI-QDA standard",
    fr: "Standard REFI-QDA", it: "Standard REFI-QDA" },
  "st.bias": { de: "BIAS.City", en: "BIAS.City", fr: "BIAS.City",
    it: "BIAS.City" },
  // Der Institutsname bleibt in ALLEN Sprachen deutsch — Eigenname,
  // keine erfundene Amtsübersetzung.
  "st.bias.sub": { de: "Basel Institut für angewandte Stadtforschung",
    en: "Basel Institut für angewandte Stadtforschung",
    fr: "Basel Institut für angewandte Stadtforschung",
    it: "Basel Institut für angewandte Stadtforschung" },
  "st.bias.text": {
    de: "LocalTranscript entsteht am B-IAS. Die App ist freie Software und bleibt es.",
    en: "LocalTranscript is made at B-IAS. The app is free software and stays that way.",
    fr: "LocalTranscript est développé au B-IAS. L'application est un logiciel libre et le reste.",
    it: "LocalTranscript nasce al B-IAS. L'app è software libero e tale resta." },
  "st.bias.link": { de: "bias.city öffnen", en: "Open bias.city",
    fr: "Ouvrir bias.city", it: "Apri bias.city" },

  // Kit-Bausteine (components/ui.tsx) — enrich nutzt denselben
  // ui.*-Namensraum; ui.caret.* sind buchstäblich dieselben Schlüssel.
  "ui.caret.aufklappen": { de: "Aufklappen", en: "Expand",
    fr: "Déplier", it: "Espandi" },
  "ui.caret.zuklappen": { de: "Zuklappen", en: "Collapse",
    fr: "Replier", it: "Comprimi" },
  "ui.panel.einblenden": { de: "Einblenden", en: "Show",
    fr: "Afficher", it: "Mostra" },
  "ui.panel.ausblenden": { de: "Ausblenden", en: "Hide",
    fr: "Masquer", it: "Nascondi" },
  "ui.tab.schliessen": { de: "Tab schließen", en: "Close tab",
    fr: "Fermer l'onglet", it: "Chiudi scheda" },
  "ui.tab.schliessen.x": { de: "{t} schließen", en: "Close {t}",
    fr: "Fermer {t}", it: "Chiudi {t}" },
  "ui.tabelle.alle": { de: "Alle sichtbaren an/abwählen",
    en: "Select/deselect all visible",
    fr: "Tout sélectionner/désélectionner",
    it: "Seleziona/deseleziona tutti i visibili" },
  "ui.tabelle.zeile": { de: "Zeile auswählen", en: "Select row",
    fr: "Sélectionner la ligne", it: "Seleziona la riga" },
  "ui.zuruecksetzen": { de: "Zurücksetzen", en: "Reset",
    fr: "Réinitialiser", it: "Reimposta" },

  // Einstellungen
  "st.speicherort": { de: "Speicherort", en: "Storage", fr: "Emplacement",
    it: "Posizione" },
  "st.speicherort.text": {
    de: "Bibliotheks-Ordner — je Transkript ein Unterordner.",
    en: "Library folder — one subfolder per transcript.",
    fr: "Dossier bibliothèque — un sous-dossier par transcription.",
    it: "Cartella biblioteca — una sottocartella per trascrizione." },
  "st.aendern": { de: "Ändern …", en: "Change …", fr: "Modifier …",
    it: "Modifica …" },
  "st.standards": { de: "Standard-Optionen", en: "Default options",
    fr: "Options par défaut", it: "Opzioni predefinite" },
  "st.uisprache": { de: "Oberflächen-Sprache", en: "Interface language",
    fr: "Langue de l'interface", it: "Lingua dell'interfaccia" },
  "st.datenschutz": { de: "Datenschutz", en: "Privacy",
    fr: "Confidentialité", it: "Privacy" },
  "st.datenschutz.text": {
    de: "Läuft vollständig lokal (nur 127.0.0.1) — keine Cloud, keine Netzwerk-Übertragung. Audio und Transkripte liegen ausschließlich im Bibliotheks-Ordner, bis du sie löschst (Papierkorb der Bibliothek statt Löschen); temporäre Arbeitsdateien werden direkt nach jedem Lauf entfernt.",
    en: "Runs fully local (127.0.0.1 only) — no cloud, no network transfer. Audio and transcripts live solely in the library folder until you delete them (library trash instead of deletion); temporary working files are removed right after each run.",
    fr: "Fonctionne entièrement en local (127.0.0.1 uniquement) — pas de cloud, aucun transfert réseau. Audio et transcriptions restent dans le dossier bibliothèque jusqu'à leur suppression (corbeille de la bibliothèque) ; les fichiers temporaires sont supprimés après chaque traitement.",
    it: "Funziona completamente in locale (solo 127.0.0.1) — nessun cloud, nessun trasferimento di rete. Audio e trascrizioni restano nella cartella biblioteca finché non li elimini (cestino della biblioteca); i file temporanei vengono rimossi subito dopo ogni elaborazione." },
  "st.lizenzen": { de: "Lizenzen", en: "Licenses", fr: "Licences",
    it: "Licenze" },
  "st.lizenzen.text": {
    de: "whisper.cpp (MIT) · Modell large-v3-turbo (OpenAI, MIT) · silero-vad (MIT) · SpeechBrain ECAPA (Apache-2.0) · PyMuPDF fürs Dossier-PDF (AGPL-3.0) · Recursive-Schrift im enrich-Export (SIL OFL 1.1) · FastAPI/uvicorn (MIT) · React/Radix (MIT) · Lucide (ISC) · ffmpeg (GPL-Build) — LocalTranscript selbst: GPL-3.0-or-later.",
    en: "whisper.cpp (MIT) · large-v3-turbo model (OpenAI, MIT) · silero-vad (MIT) · SpeechBrain ECAPA (Apache-2.0) · PyMuPDF for the dossier PDF (AGPL-3.0) · Recursive typeface in enrich export (SIL OFL 1.1) · FastAPI/uvicorn (MIT) · React/Radix (MIT) · Lucide (ISC) · ffmpeg (GPL build) — LocalTranscript itself: GPL-3.0-or-later.",
    fr: "whisper.cpp (MIT) · modèle large-v3-turbo (OpenAI, MIT) · silero-vad (MIT) · SpeechBrain ECAPA (Apache-2.0) · PyMuPDF pour le PDF du dossier (AGPL-3.0) · police Recursive dans l'export enrich (SIL OFL 1.1) · FastAPI/uvicorn (MIT) · React/Radix (MIT) · Lucide (ISC) · ffmpeg (build GPL) — LocalTranscript : GPL-3.0-or-later.",
    it: "whisper.cpp (MIT) · modello large-v3-turbo (OpenAI, MIT) · silero-vad (MIT) · SpeechBrain ECAPA (Apache-2.0) · PyMuPDF per il PDF del dossier (AGPL-3.0) · carattere Recursive nell'export enrich (SIL OFL 1.1) · FastAPI/uvicorn (MIT) · React/Radix (MIT) · Lucide (ISC) · ffmpeg (build GPL) — LocalTranscript: GPL-3.0-or-later." },
  "st.app": { de: "LocalTranscript", en: "LocalTranscript",
    fr: "LocalTranscript", it: "LocalTranscript" },
  "st.app.sub": {
    de: "Freie Software — GPL-3.0-or-later · BIAS.City",
    en: "Free software — GPL-3.0-or-later · BIAS.City",
    fr: "Logiciel libre — GPL-3.0-or-later · BIAS.City",
    it: "Software libero — GPL-3.0-or-later · BIAS.City" },
  "st.app.text": {
    de: "Quellcode, Lizenztext und Releases auf GitHub. Die App übernimmt die GPL bewusst als eigene Lizenz — das strengste mitgelieferte Werkzeug (ffmpeg, GPL-Build) setzt den Takt.",
    en: "Source code, license text and releases on GitHub. The app deliberately adopts the GPL as its own license — the strictest bundled tool (ffmpeg, GPL build) sets the pace.",
    fr: "Code source, licence et versions sur GitHub. L'application adopte délibérément la GPL — l'outil embarqué le plus strict (ffmpeg, build GPL) donne le ton.",
    it: "Codice sorgente, licenza e release su GitHub. L'app adotta deliberatamente la GPL — lo strumento incluso più restrittivo (ffmpeg, build GPL) detta il passo." },
  "ueber.titel": { de: "Über LocalTranscript",
    en: "About LocalTranscript", fr: "À propos de LocalTranscript",
    it: "Informazioni su LocalTranscript" },
  "ueber.version": { de: "Version {v}", en: "Version {v}",
    fr: "Version {v}", it: "Versione {v}" },
  "ueber.herkunft": {
    de: "LocalTranscript entsteht am B/IAS — Basel Institut für angewandte Stadtforschung, BIAS.City.",
    en: "LocalTranscript is built at B/IAS — Basel Institut für angewandte Stadtforschung, BIAS.City.",
    fr: "LocalTranscript est développé au B/IAS — Basel Institut für angewandte Stadtforschung, BIAS.City.",
    it: "LocalTranscript nasce al B/IAS — Basel Institut für angewandte Stadtforschung, BIAS.City." },
  "st.link.lizenztext": { de: "Lizenztext (GPL-3.0)",
    en: "License text (GPL-3.0)", fr: "Texte de licence (GPL-3.0)",
    it: "Testo della licenza (GPL-3.0)" },
  "allg.schliessen": { de: "Schließen", en: "Close", fr: "Fermer",
    it: "Chiudi" },
  "st.link.repo": { de: "Quellcode (GitHub)", en: "Source code (GitHub)",
    fr: "Code source (GitHub)", it: "Codice sorgente (GitHub)" },
  "st.link.releases": { de: "Releases", en: "Releases",
    fr: "Versions", it: "Release" },
  "st.link.ffmpegbuild": { de: "ffmpeg-Build (martin-riedl.de)",
    en: "ffmpeg build (martin-riedl.de)",
    fr: "Build ffmpeg (martin-riedl.de)",
    it: "Build ffmpeg (martin-riedl.de)" },
  "st.link.ffmpegsrc": { de: "ffmpeg-Quellcode", en: "ffmpeg source",
    fr: "Source ffmpeg", it: "Sorgente ffmpeg" },
  "st.modelle": { de: "Modelle in {d}", en: "Models in {d}",
    fr: "Modèles dans {d}", it: "Modelli in {d}" },

  "allg.ok": { de: "OK", en: "OK", fr: "OK", it: "OK" },
  "allg.abbrechen": { de: "Abbrechen", en: "Cancel", fr: "Annuler",
    it: "Annulla" },
  "allg.name": { de: "Name", en: "Name", fr: "Nom", it: "Nome" },
  "allg.fehler": { de: "Fehler: {e}", en: "Error: {e}",
    fr: "Erreur : {e}", it: "Errore: {e}" },
};

export function uebersetze(key: string, sprache: Sprache,
    params?: Record<string, string | number>): string {
  const e = W[key];
  let text = e ? (e[sprache] ?? e.en) : key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      text = text.replace(`{${k}}`, String(v));
    }
  }
  return text;
}

export function useT(): (key: string,
    params?: Record<string, string | number>) => string {
  const s = useSprache();
  return (key, params) => uebersetze(key, s, params);
}

/** Job-Statuszeile aus den neutralen Backend-Tokens übersetzen. */
export function jobText(tr: (k: string,
    p?: Record<string, string | number>) => string,
    status: string, message: string): string {
  if (message.startsWith("transkribiere:")) {
    const [a, b] = message.slice(14).split("/");
    return tr("job.transkribiere.n", { a, b });
  }
  const map: Record<string, string> = {
    konvertiere: "job.konvertiere", sprecher: "job.sprecher",
    transkribiere: "job.transkribiere", speichere: "job.speichere",
    fertig: "job.fertig", abgebrochen: "job.abgebrochen",
    fehler: "job.fehler",
  };
  if (map[message]) return tr(map[message]);
  if (status === "pending") return tr("job.warte");
  return message;
}
