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
  "ed.play": { de: "Abspielen", en: "Play", fr: "Lecture",
    it: "Riproduci" },
  "ed.pause": { de: "Pause", en: "Pause", fr: "Pause", it: "Pausa" },
  "ed.loop": { de: "Segment wiederholen", en: "Loop segment",
    fr: "Boucler le segment", it: "Ripeti segmento" },
  "ed.rueck5": { de: "5 s zurück (Ctrl+←)", en: "5 s back (Ctrl+←)",
    fr: "5 s en arrière (Ctrl+←)", it: "5 s indietro (Ctrl+←)" },
  "ed.vor5": { de: "5 s vor (Ctrl+→)", en: "5 s forward (Ctrl+→)",
    fr: "5 s en avant (Ctrl+→)", it: "5 s avanti (Ctrl+→)" },
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
    de: "whisper.cpp (MIT) · Modell large-v3-turbo (OpenAI, MIT) · silero-vad (MIT) · SpeechBrain ECAPA (Apache-2.0) · Recursive-Schrift im enrich-Export (SIL OFL 1.1) · FastAPI/uvicorn (MIT) · React/Radix (MIT) · Lucide (ISC) · ffmpeg (GPL-Build) — LocalTranscript selbst: GPL-3.0-or-later.",
    en: "whisper.cpp (MIT) · large-v3-turbo model (OpenAI, MIT) · silero-vad (MIT) · SpeechBrain ECAPA (Apache-2.0) · Recursive typeface in enrich export (SIL OFL 1.1) · FastAPI/uvicorn (MIT) · React/Radix (MIT) · Lucide (ISC) · ffmpeg (GPL build) — LocalTranscript itself: GPL-3.0-or-later.",
    fr: "whisper.cpp (MIT) · modèle large-v3-turbo (OpenAI, MIT) · silero-vad (MIT) · SpeechBrain ECAPA (Apache-2.0) · police Recursive dans l'export enrich (SIL OFL 1.1) · FastAPI/uvicorn (MIT) · React/Radix (MIT) · Lucide (ISC) · ffmpeg (build GPL) — LocalTranscript : GPL-3.0-or-later.",
    it: "whisper.cpp (MIT) · modello large-v3-turbo (OpenAI, MIT) · silero-vad (MIT) · SpeechBrain ECAPA (Apache-2.0) · carattere Recursive nell'export enrich (SIL OFL 1.1) · FastAPI/uvicorn (MIT) · React/Radix (MIT) · Lucide (ISC) · ffmpeg (build GPL) — LocalTranscript: GPL-3.0-or-later." },
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
