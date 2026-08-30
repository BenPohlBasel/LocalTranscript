// Sichere sessionStorage-Zugriffe (Prüfbericht #21/#51): Safari Private
// Mode / Quota werfen — UI darf daran nie sterben. Schlüssel zentral.
export const KEYS = {
  hubTabs: "enrich.hub.tabs",
  viewerLeft: "enrich.viewer.left",
  viewerRight: "enrich.viewer.right",
  viewerHidden: "enrich.viewer.hidden",
  viewerEdits: (project: string, dossier: string) =>
    `enrich.viewer.edits.${project}/${dossier}`,
  // Persistente Arbeits-Zustände (User 2026-07-31): Tab-Wechsel und
  // Reload verlieren Suche/Schreiben/Fragen nicht mehr.
  searchState: "enrich.search.state",
  // Kachel-Reihenfolge des Hubs (localStorage — überlebt Sessions)
  hubOrder: "enrich.hub.order",
  // Sorten-Filter (localStorage, User 2026-08-06): Liste der AUS-
  // geblendeten Sorten — JE ANSICHT eigener Zustand (liste|index)
  sortenAus: (view: string) => `enrich.studio.sorten-aus.${view}`,
  // Register: Sortierung über den ganzen Index (alphabet|haeufung|art)
  indexSort: "enrich.studio.index-sort",
  // Studio: offene Collapsibles + Scroll-Position je Ansicht+Dossier
  // (sessionStorage — überleben den Subtab-Wechsel, User 2026-08-06)
  studioOffen: (view: string, dossier: string) =>
    `enrich.studio.offen.${view}.${dossier}`,
  studioScroll: (view: string, dossier: string) =>
    `enrich.studio.scroll.${view}.${dossier}`,
  //: aktiver Studio-Subtab (User 2026-08-20: Pulldown statt Leiste —
  //: der Wechsel zwischen Dossiers/Tabs behält die Ansicht)
  studioView: "enrich.studio.view",
  writeSlug: (project: string) => `enrich.write.slug.${project}`,
  chatTurns: (project: string) => `enrich.chat.turns.${project}`,
  //: Server-ID des laufenden Chats (Audit 2026-08-22 #2: ohne sie legte
  //: der Auto-Save nach jedem Remount ein DUPLIKAT an)
  chatId: (project: string) => `enrich.chat.id.${project}`,
  //: Scope des Finden-Tabs (Dossiers + Sorten-Whitelist, localStorage —
  //: ersetzt chatDossiers; Audit #42: Schlüssel gehören in die Registry)
  findenScope: (project: string) => `enrich.finden.scope.${project}`,
  //: aktive Aktion des Finden-Tabs (suchen | fragen)
  findenAktion: "enrich.finden.aktion",
  //: Sidebar-Sub-Tab in Finden (Integration 2026-08-23): chats | schreiben
  findenSeite: "enrich.finden.seite",
  //: Ast-Sidebar im Fragen-Tab (User 2026-08-09): Pin + Verlaufs-Chips
  //: überleben Tab-Wechsel und Link-Sprünge
  chatPin: (project: string) => `enrich.chat.pin.${project}`,
  chatPinVerlauf: (project: string) => `enrich.chat.pinverlauf.${project}`,
  //: Schnell-Kodierer (User 2026-08-16): zuletzt benutzte Code-IDs je
  //: Projekt (localStorage — Ziffern 1–9 sollen Sessions überleben)
  ucodesZuletzt: (project: string) => `enrich.usercodes.zuletzt.${project}`,
  codebuchDossier: (project: string) => `enrich.codebuch.dossier.${project}`,
} as const;

export function sget(key: string): string | null {
  try { return sessionStorage.getItem(key); } catch { return null; }
}

export function sset(key: string, value: string): void {
  try { sessionStorage.setItem(key, value); } catch { /* Quota/Private Mode */ }
}

export function sremove(key: string): void {
  try { sessionStorage.removeItem(key); } catch { /* ignorieren */ }
}

export function sgetJson<T>(key: string, fallback: T): T {
  const raw = sget(key);
  if (raw === null) return fallback;
  try { return JSON.parse(raw) as T; } catch { return fallback; }
}

// localStorage: Darstellungs-Präferenzen, die Fenster UND Sessions
// überleben sollen (z. B. Icons an/aus) — gleiche Absicherung.
export function lget(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}

export function lset(key: string, value: string): void {
  try { localStorage.setItem(key, value); } catch { /* Private Mode */ }
}
