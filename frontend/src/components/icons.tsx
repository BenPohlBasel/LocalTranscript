// Icon-Abstraktion (User-Vorgabe): Module referenzieren nur NAMEN — die
// Bibliothek (Lucide, linienbasiert) ist ausschließlich HIER bekannt und
// damit austauschbar (Tabler/Radix Icons = nur diese Datei ändern).
// Nur-Text-Modus: iconsEnabled() aus → Icon rendert nichts; alle
// Bedienelemente tragen ohnehin Text-Labels.
import {
  AlignJustify, AlignLeft, ClipboardCheck, Columns2, Database, Download,
  FileInput, FileText, Files, FolderOpen, Hammer, Highlighter,
  House, Import, LayoutGrid, Library, ListChecks, Lock, LockOpen,
  Bold, Italic, List as ListIcon,
  MessageCircleQuestionMark, Minus, PenLine, Pencil, Play, Quote, Search,
  Settings, Tags, Waypoints, type LucideIcon,
} from "lucide-react";
import { lget, lset } from "../lib/storage";

export type IconName =
  | "hub" | "werkstatt" | "import" | "runs" | "projects" | "settings"
  | "review" | "dossiers" | "dossier-import" | "search" | "viewer" | "write"
  | "library" | "draw" | "home" | "analysis" | "relational"
  | "studio" | "download" | "batch" | "text" | "index" | "cite"
  | "lock" | "lock-open" | "chat"
  | "text-wenig" | "text-mittel" | "text-viel" | "kodieren"
  | "fett" | "kursiv" | "liste" | "zitat";

const ICONS: Record<IconName, LucideIcon> = {
  kodieren: Highlighter,
  "text-wenig": Minus,
  "text-mittel": AlignLeft,
  "text-viel": AlignJustify,
  hub: LayoutGrid,
  home: House,
  werkstatt: Hammer,
  import: Import,
  runs: Play,
  projects: FolderOpen,
  settings: Settings,
  review: ClipboardCheck,
  dossiers: Files,
  "dossier-import": FileInput,
  search: Search,
  viewer: Highlighter,
  write: PenLine,
  library: Library,
  draw: Pencil,
  analysis: Tags,
  relational: Waypoints,
  studio: Columns2,
  download: Download,
  batch: ListChecks,
  text: FileText,
  index: Database,
  cite: Quote,
  fett: Bold,
  kursiv: Italic,
  liste: ListIcon,
  zitat: Quote,
  lock: Lock,
  "lock-open": LockOpen,
  chat: MessageCircleQuestionMark,
};

const ICONS_KEY = "enrich.appearance.icons";

/** Nur-Text-Modus (User-Vorgabe): Appearance überlebt Fenster & Sessions. */
export function iconsEnabled(): boolean {
  return lget(ICONS_KEY) !== "0";
}

export function setIconsEnabled(on: boolean): void {
  lset(ICONS_KEY, on ? "1" : "0");
}

export function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  if (!iconsEnabled()) return null;
  const C = ICONS[name];
  if (!C) return null;
  return <C size={size} strokeWidth={1.75} aria-hidden
            style={{ flexShrink: 0, verticalAlign: "-2px" }} />;
}
