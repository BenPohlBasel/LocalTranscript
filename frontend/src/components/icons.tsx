// Icon-Abstraktion (enrich-Kit-Muster): Module referenzieren nur NAMEN —
// die Bibliothek (Lucide) ist ausschließlich HIER bekannt.
import {
  ChevronLeft, Download, FastForward, FileText, FolderOpen, Import,
  Library, Pause, Pencil, Play, Plus, Repeat, Rewind, Scissors,
  Settings, Trash2, Users, Volume2, X, type LucideIcon,
} from "lucide-react";

export type IconName =
  | "library" | "settings" | "play" | "pause" | "download" | "import"
  | "text" | "edit" | "trash" | "split" | "speakers" | "plus"
  | "back" | "folder" | "close" | "sample" | "rewind" | "forward"
  | "loop";

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
  split: Scissors,
  speakers: Users,
  plus: Plus,
  back: ChevronLeft,
  folder: FolderOpen,
  close: X,
  sample: Volume2,
  rewind: Rewind,
  forward: FastForward,
  loop: Repeat,
};

export function iconsEnabled(): boolean { return true; }

export function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  const C = ICONS[name];
  if (!C) return null;
  return <C size={size} strokeWidth={1.75} aria-hidden
            style={{ flexShrink: 0, verticalAlign: "-2px" }} />;
}
