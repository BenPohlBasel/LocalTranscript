// Icon-Abstraktion (enrich-Kit-Muster): Module referenzieren nur NAMEN —
// die Bibliothek (Lucide) ist ausschließlich HIER bekannt.
// Gleicher Satz wie enrich: Lucide, linienbasiert, strokeWidth 1.75,
// Größenleiter 12/14/16/18 (Anzeige 24/40). „split"/„merge" sind ein
// Paar (Linie teilt sich ⇄ Linien laufen zusammen) — die Schere fiel
// raus, sie war das einzige Gegenstands-Icon im Satz (2026-09-09).
import {
  ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight,
  Download, FileText,
  FolderOpen, Import, Library, Merge, Pause, Pencil, Play, Plus,
  Repeat, Search, Settings, Split, Trash2, Users, Volume2, X,
  type LucideIcon,
} from "lucide-react";

export type IconName =
  | "library" | "settings" | "play" | "pause" | "download" | "import"
  | "text" | "edit" | "trash" | "split" | "speakers" | "plus"
  | "back" | "folder" | "close" | "sample" | "rewind" | "forward"
  | "loop" | "merge" | "search" | "next";

const ICONS: Record<IconName, LucideIcon> = {
  library: Library,
  settings: Settings,
  play: Play,
  pause: Pause,
  download: Download,
  import: Import,
  text: FileText,
  edit: Pencil,
  trash: Trash2,
  split: Split,
  speakers: Users,
  plus: Plus,
  back: ChevronLeft,
  next: ChevronRight,
  folder: FolderOpen,
  close: X,
  sample: Volume2,
  rewind: ChevronsLeft,
  forward: ChevronsRight,
  loop: Repeat,
  merge: Merge,
  search: Search,
};

export function iconsEnabled(): boolean { return true; }

export function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  const C = ICONS[name];
  if (!C) return null;
  return <C size={size} strokeWidth={1.75} aria-hidden
            style={{ flexShrink: 0, verticalAlign: "-2px" }} />;
}
