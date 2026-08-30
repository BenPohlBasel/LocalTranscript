// API-Typen + Fetch-Helfer — EINE Stelle (enrich-Kit-Regel).
import { isTauri } from "./tauri";

export const BACKEND_PORT = 44100;
export const API_BASE = isTauri()
  ? `http://127.0.0.1:${BACKEND_PORT}` : "";

export type Sprecher = { id: string; name: string };
export type Segment = {
  id: string; start: number; end: number;
  sprecher: string | null; text: string;
};
export type Transkript = {
  schema: number; id: string; name: string; created: string;
  updated: string; audio: string | null;
  quelle: Record<string, unknown>;
  sprecher: Sprecher[]; segmente: Segment[];
};
export type EintragMeta = {
  id: string; name: string; created: string; updated: string;
  dauer: number; segmente: number; sprecher: number; audio: boolean;
  quelle: Record<string, unknown>;
};
export type Job = {
  id: string; filename: string; status: string; progress: number;
  message: string; partial_text: string; error: string | null;
  eintrag: string | null; created_at: string;
  params: Record<string, unknown>;
};
export type Settings = {
  library_root: string; default_library_root: string; model: string;
  language: string; diarize: boolean; speaker_range: string;
  cluster_threshold: number; ui_language: string;
};
export type ModellInfo = { name: string; size_mb: number };

export function errMsg(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

async function _check(r: Response): Promise<Response> {
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const j = await r.json();
      if (j && typeof j.detail === "string") detail = j.detail;
    } catch { /* Klartext reicht */ }
    throw new Error(detail);
  }
  return r;
}

export async function apiGet<T>(pfad: string): Promise<T> {
  const r = await _check(await fetch(`${API_BASE}${pfad}`));
  return r.json() as Promise<T>;
}

export async function apiSend<T>(pfad: string, body?: unknown,
                                 method = "POST"): Promise<T> {
  const r = await _check(await fetch(`${API_BASE}${pfad}`, {
    method,
    headers: body === undefined ? undefined
      : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  }));
  return r.json() as Promise<T>;
}

export async function apiUpload<T>(pfad: string,
                                   form: FormData): Promise<T> {
  const r = await _check(await fetch(`${API_BASE}${pfad}`,
                                     { method: "POST", body: form }));
  return r.json() as Promise<T>;
}

/** Farbpalette für Sprecher-Chips (Radix-Badge-Farben, Index = stabil
    über die Entitäts-Reihenfolge). */
export const SPRECHER_FARBEN = [
  "indigo", "amber", "green", "crimson", "teal", "violet", "orange",
  "cyan", "pink", "lime",
] as const;

export function sprecherFarbe(sprecher: Sprecher[], sid: string | null):
    (typeof SPRECHER_FARBEN)[number] | "gray" {
  if (!sid) return "gray";
  const i = sprecher.findIndex((s) => s.id === sid);
  return i < 0 ? "gray" : SPRECHER_FARBEN[i % SPRECHER_FARBEN.length];
}

/** IMMER hh:mm:ss (User-Regel). */
export function hms(sekunden: number): string {
  const s = Math.max(0, Math.floor(sekunden));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${
    String(s % 60).padStart(2, "0")}`;
}
