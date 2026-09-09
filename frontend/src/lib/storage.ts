// Sichere Storage-Zugriffe (enrich-Kit-Muster): Private Mode / Quota
// werfen — das UI darf daran nie sterben. Schlüssel zentral.
export const KEYS = {
  sprache: "lt.ui.sprache",
  optionenOffen: "lt.bibliothek.optionen",
  editorFolgen: "lt.editor.folgen",
  editorSpeed: "lt.editor.speed",
  sidebarSprecher: "lt.editor.sprecher-panel",
  editorSeitenTab: "lt.editor.seitentab",
} as const;

export function sget(key: string): string | null {
  try { return sessionStorage.getItem(key); } catch { return null; }
}
export function sset(key: string, value: string): void {
  try { sessionStorage.setItem(key, value); } catch { /* Quota */ }
}
export function lget(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
export function lset(key: string, value: string): void {
  try { localStorage.setItem(key, value); } catch { /* Private Mode */ }
}
