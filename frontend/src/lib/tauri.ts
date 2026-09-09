// Tauri-Brücke: Feature-Detection + die wenigen nativen Wege (Backend
// starten, Save-/Open-Dialoge, Ordner öffnen). Die Shell ist dumm —
// alle Daten besitzt das Backend.
export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/** Backend über das Rust-Kommando starten; liefert erst zurück, wenn
    /api/health antwortet (oder wirft mit Fehlertext). */
export async function backendStarten(): Promise<void> {
  if (!isTauri()) return;
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("backend_starten");
}

export async function savePath(defaultName: string,
                               ext: string): Promise<string | null> {
  const { save } = await import("@tauri-apps/plugin-dialog");
  return save({
    defaultPath: defaultName,
    filters: [{ name: ext.toUpperCase(), extensions: [ext] }],
  });
}

export async function pickAudio(title?: string,
                                multiple = true):
    Promise<string[] | null> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const r = await open({ multiple, title, filters: [{
    name: "Audio",
    extensions: ["mp3", "m4a", "aac", "wav", "ogg", "flac", "webm"] }] });
  if (r == null) return null;
  return Array.isArray(r) ? r : [r];
}

export async function pickTranskript(): Promise<string | null> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const r = await open({ multiple: false, filters: [{
    name: "Transkript", extensions: ["vtt", "webvtt", "csv"] }] });
  return typeof r === "string" ? r : null;
}

export async function pickOrdner(): Promise<string | null> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const r = await open({ directory: true, multiple: false });
  return typeof r === "string" ? r : null;
}

export async function ordnerOeffnen(pfad: string): Promise<void> {
  if (!isTauri()) return;
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("ordner_oeffnen", { pfad });
}

/** Nativer Datei-Drop (Tauri fängt HTML5-DnD ab und liefert PFADE —
    genau richtig: große Audios laufen nie durch HTTP). */
export async function onFileDrop(
  cb: (paths: string[]) => void): Promise<() => void> {
  if (!isTauri()) return () => {};
  const { getCurrentWebview } = await import("@tauri-apps/api/webview");
  return getCurrentWebview().onDragDropEvent((e) => {
    if (e.payload.type === "drop") cb(e.payload.paths);
  });
}

/** „About LocalTranscript" aus dem Menü — die Shell schickt nur das
    Signal, den Dialog baut das Frontend (übersetzt, mit Links). */
export async function onUeber(cb: () => void): Promise<() => void> {
  if (!isTauri()) return () => {};
  const { listen } = await import("@tauri-apps/api/event");
  return listen("ueber", () => cb());
}
