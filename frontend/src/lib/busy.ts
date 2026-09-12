// App-weiter Beschäftigt-Zähler (User 2026-09-12): jede laufende
// Backend-Anfrage und jeder grosse Aufbau (Editor mit vielen Zeilen)
// meldet sich an — die Kopfzeile zeigt derweil ein drehendes Rad.
// Ein Zähler, kein Flag: zwei parallele Anfragen dürfen sich nicht
// gegenseitig das Rad abschalten.
import { useSyncExternalStore } from "react";

let anzahl = 0;
const hoerer = new Set<() => void>();
const melde = () => { for (const h of hoerer) h(); };

export function beginne(): void { anzahl += 1; melde(); }
export function ende(): void { anzahl = Math.max(0, anzahl - 1); melde(); }

/** Läuft gerade etwas? (Reaktiv, für den Indikator.) */
export function useBusy(): boolean {
  return useSyncExternalStore(
    (h) => { hoerer.add(h); return () => { hoerer.delete(h); }; },
    () => anzahl > 0);
}
