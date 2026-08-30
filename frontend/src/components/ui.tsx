// Wiederverwendbare UI-Bausteine (einzige Stelle, die die Komponenten-
// Bibliothek kennt — Radix Themes, Apple-naher Look). Module importieren
// NUR von hier; die Bibliothek bleibt austauschbar.
import {
  Badge, Box, Button, Callout, Checkbox, Dialog, Flex, Heading, HoverCard,
  ScrollArea, SegmentedControl, Select, Slider, Table, Text, TextField,
} from "@radix-ui/themes";
import {
  createContext, useContext, useEffect, useId, useLayoutEffect, useMemo,
  useRef, useState,
  type CSSProperties,
  type ComponentProps, type MouseEvent as ReactMouseEvent, type ReactNode,
  type RefObject,
} from "react";
import { sget, sset } from "../lib/storage";
import { Icon, type IconName } from "./icons";

/** Von der Shell bereitgestellte Kopfzeilen-Elemente (Ablösen/Andocken,
    Status) — das Panel rendert sie in SEINER Kopfzeile mit. So gibt es
    genau EINE Kopfzeile pro Modul statt Card-in-Card. */
export const ShellTrailer = createContext<ReactNode>(null);

/** Sub-Navigation eines Komposit-Moduls (z. B. Werkstatt: Projekte/Import/…):
    erscheint LINKS in der Panel-Kopfzeile, vor dem Titel — weiterhin genau
    eine Kopfzeile. */
export const SubNavSlot = createContext<ReactNode>(null);

/** Globaler Fußzeilen-Auftakt (App-Rahmen → jedes Panel): erscheint LINKS
    in JEDER Panel-Fußzeile — aktives Projekt + Status-Ampeln; die Fußzeile
    ist der etablierte Ort für Kontext und Status (User 2026-08-02). */
export const FootLead = createContext<ReactNode>(null);

export { Badge, Box, Button, Checkbox, Flex, Grid, Heading, Progress, Select,
         Switch, Text, TextField } from "@radix-ui/themes";

// ---------- Modal-Dialog (Kit-Baustein, z. B. Pfad-Auswahl) ----------

export function ModalDialog({ open, onOpenChange, title, children, footer,
                              width = 560, fullscreen }: {
  open: boolean; onOpenChange: (open: boolean) => void; title: ReactNode;
  children: ReactNode; footer?: ReactNode; width?: number;
  /** Fensterfüllend (Werkstatt-Import 2026-08-13): 96vw×92vh, der Body
      wird flex-Spalte OHNE eigenen Scroll — die Kinder besitzen ihn
      (Muster ZoteroImportInhalt: die Tabelle scrollt in ihrer Box). */
  fullscreen?: boolean;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Content style={fullscreen
        ? { maxWidth: "96vw", width: "96vw", height: "92vh",
            display: "flex", flexDirection: "column" }
        : { maxWidth: width }}>
        <Dialog.Title>{title}</Dialog.Title>
        <Box style={fullscreen
          ? { flex: 1, minHeight: 0, display: "flex",
              flexDirection: "column", overflow: "hidden" }
          : { maxHeight: "55vh", overflowY: "auto" }}>{children}</Box>
        {footer && <Flex gap="2" mt="3" justify="end" align="center">{footer}</Flex>}
      </Dialog.Content>
    </Dialog.Root>
  );
}

// ---------- Panel: Kopf / scrollender Körper / Fußleiste ----------

export function Panel({ title, actions, footer, children, fill }: {
  title?: ReactNode; actions?: ReactNode; footer?: ReactNode; children: ReactNode;
  /** true: Körper scrollt NICHT selbst — das Modul verwaltet den eigenen
      Scroll-Viewport (z. B. der Seiten-Strom des Viewers). */
  fill?: boolean;
}) {
  const trailer = useContext(ShellTrailer);
  const subnav = useContext(SubNavSlot);
  const lead = useContext(FootLead);
  return (
    <Flex direction="column" className="ui-panel">
      {(title || actions || trailer || subnav) && (
        <Flex align="center" gap="3" className="ui-panel-head" wrap="wrap">
          {/* Titel VOR der Sub-Navigation (User 2026-08-02) */}
          {title && <Heading size="3">{title}</Heading>}
          {subnav}
          <Box flexGrow="1" />
          {actions}
          {trailer}
        </Flex>
      )}
      {fill
        ? <Flex direction="column" className="ui-panel-body" overflow="hidden">{children}</Flex>
        : <ScrollArea className="ui-panel-body" scrollbars="vertical">{children}</ScrollArea>}
      {(lead || footer) && (
        <Flex align="center" gap="3" className="ui-panel-foot" wrap="wrap">
          {lead}
          {footer}
        </Flex>
      )}
    </Flex>
  );
}

/** Einstellungs-Karte (User 2026-08-15, „Layout-Chaos"): logisch
    zusammenhängende Einheit mit Überschrift + Subline, fließt im
    responsiven Grid (1–2 Spalten). `breit` spannt über alle Spalten. */
export function Karte({ titel, subline, breit, children }: {
  titel: ReactNode; subline?: ReactNode; breit?: boolean;
  children: ReactNode;
}) {
  return (
    <Box className="ui-karte"
         style={breit ? { gridColumn: "1 / -1" } : undefined}>
      <Heading size="2" mb={subline ? "0" : "3"}>{titel}</Heading>
      {subline && (
        <Text size="1" color="gray" as="div" mb="3">{subline}</Text>
      )}
      {children}
    </Box>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <Flex align="center" justify="center" p="8" style={{ minHeight: 160 }}>
      {/* as="div": Aufrufer übergeben auch Block-Elemente (Prüfbericht #39) */}
      <Text as="div" align="center" color="gray" size="2">{children}</Text>
    </Flex>
  );
}

export function ActivityNote({ children }: { children: ReactNode }) {
  // Aktivitätszeile (User 2026-07-31): pulsierender Punkt + aktueller
  // Hintergrund-Schritt (Serve /activity) — zeigt, WAS gerade passiert
  return (
    <Flex gap="2" align="center" px="4" py="1">
      <span className="ui-activity-dot" />
      <Text size="1" color="gray">{children}</Text>
    </Flex>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <Callout.Root color="red" size="1" m="3">
      <Callout.Text>{children}</Callout.Text>
    </Callout.Root>
  );
}

export function SuccessNote({ children }: { children: ReactNode }) {
  return (
    <Callout.Root color="green" size="1" m="3">
      <Callout.Text>{children}</Callout.Text>
    </Callout.Root>
  );
}

/** Abschnitts-Kopf innerhalb eines Panels. */
export function SectionHead({ children }: { children: ReactNode }) {
  return (
    <Box px="4" py="2" style={{ background: "var(--gray-a2)",
                                borderBottom: "1px solid var(--gray-a4)" }}>
      <Text size="1" weight="medium" color="gray">{children}</Text>
    </Box>
  );
}

/** Halbtorten-Anzeige (User 2026-07-28: Rechner-Status in den Einstellungen).
    percent 0–100; Farbe kippt bei 75/90 auf amber/rot. */
export function Gauge({ label, percent, detail }: {
  label: string; percent: number; detail?: string;
}) {
  const p = Math.max(0, Math.min(100, percent));
  const r = 40, cx = 50, cy = 50;
  const angle = Math.PI * (1 - p / 100);  // 180° (0 %) → 0° (100 %)
  const x = cx + r * Math.cos(angle), y = cy - r * Math.sin(angle);
  const color = p >= 90 ? "var(--red-9)" : p >= 75 ? "var(--amber-9)"
    : "var(--accent-9)";
  return (
    <Flex direction="column" align="center" gap="1">
      <svg width="100" height="58" viewBox="0 0 100 58" role="img"
           aria-label={`${label}: ${Math.round(p)} %`}>
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
              fill="none" stroke="var(--gray-a4)" strokeWidth="10"
              strokeLinecap="round" />
        {p > 0 && (
          <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${x} ${y}`}
                fill="none" stroke={color} strokeWidth="10"
                strokeLinecap="round" />
        )}
        <text x={cx} y={cy - 4} textAnchor="middle"
              style={{ font: "600 15px var(--default-font-family)",
                       fill: "var(--gray-12)" }}>
          {Math.round(p)}%
        </text>
      </svg>
      <Text size="1" weight="medium">{label}</Text>
      {detail && <Text size="1" color="gray">{detail}</Text>}
    </Flex>
  );
}

/** Kompaktes Schlüssel-Wert-Paar (Berichte, Metadaten). */
export function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Flex gap="2" align="center">
      <Text size="1" color="gray">{label}</Text>
      <Text size="1">{children}</Text>
    </Flex>
  );
}

// ---------- Listen ----------

export function ListRow({ selected, onClick, leading, title, meta, trailing, indent = 0 }: {
  selected?: boolean; onClick?: () => void; leading?: ReactNode;
  title: ReactNode; meta?: ReactNode; trailing?: ReactNode; indent?: number;
}) {
  return (
    <Flex align="center" gap="2" px="4" py="2"
          className={"ui-row" + (selected ? " selected" : "") + (onClick ? " clickable" : "")}
          style={indent ? { paddingLeft: 16 + indent * 18 } : undefined}
          onClick={onClick}
          // Tastaturweg für die Primärnavigation (Prüfbericht #30)
          role={onClick ? "button" : undefined}
          tabIndex={onClick ? 0 : undefined}
          onKeyDown={onClick ? (e) => {
            if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); }
          } : undefined}>
      {leading}
      <Box flexGrow="1" minWidth="0">
        <Text as="div" size="2" truncate>{title}</Text>
        {meta && <Text as="div" size="1" color="gray" truncate>{meta}</Text>}
      </Box>
      {trailing}
    </Flex>
  );
}

/** Aufklappbarer Detail-Block (Kit-Baustein, z. B. Agent-Protokoll im
    Fragen-Tab) — tastaturfähig, großes Caret, nie inline <details>. */
export function Disclosure({ label, children, defaultOpen = false,
                             open: openProp, onToggle }: {
  label: ReactNode; children: ReactNode; defaultOpen?: boolean;
  /** steuerbar (2026-08-11, User Codes): open + onToggle übernehmen */
  open?: boolean; onToggle?: (open: boolean) => void;
}) {
  const [eigen, setEigen] = useState(defaultOpen);
  const open = openProp ?? eigen;
  const setOpen = (fn: (o: boolean) => boolean) => {
    const naechster = fn(open);
    if (onToggle) onToggle(naechster);
    if (openProp === undefined) setEigen(naechster);
  };
  return (
    <Box>
      <button type="button" aria-expanded={open}
              onClick={() => setOpen((o) => !o)}
              style={{ all: "unset", cursor: "pointer",
                       display: "inline-flex", alignItems: "center",
                       gap: 4 }}>
        <Caret open={open} />
        <Text size="1" color="gray">{label}</Text>
      </button>
      {open && <Box pl="5" pt="1">{children}</Box>}
    </Box>
  );
}


/** Aufklapp-Dreieck ▸/▾ — global EIN Bauteil, bewusst groß (User-Vorgabe:
    die kleinen Dreiecke waren überall zu fummelig). */
export function Caret({ open, onClick }: {
  open: boolean; onClick?: (e: ReactMouseEvent) => void;
}) {
  if (!onClick) return <span className="ui-caret">{open ? "▾" : "▸"}</span>;
  return (
    <button type="button" className="ui-caret"
            aria-expanded={open} aria-label={open ? "Einklappen" : "Aufklappen"}
            onClick={onClick}>{open ? "▾" : "▸"}</button>
  );
}

/** Platzhalter in Baum-Zeilen ohne Kinder (hält die Einrückung). */
export function CaretLeaf() {
  return <span className="ui-caret leaf">·</span>;
}

// ---------- Epistemik-Theme (User 2026-08-06) ----------
// Drei Fragen, drei Kanäle: Beleg = Serife+Tinte («…» setzt der Aufrufer),
// LLM-Text = Rampenfarbe nach NÄHE zum Original (1 wörtlich … 4 spekulativ),
// menschlicher Text = kursiv/warmgrau. Farben und Varianten leben ZENTRAL
// im Epistemik-Block von styles.css (--epi-*) — Design-Korrekturen passieren
// nur dort, nie in den Modulen.

export type EpiKind = 1 | 2 | 3 | 4 | "quelle" | "mensch";

/** Baum-/Record-Sorte → Nähe-Stufe. EINE Wahrheit für alle Module. */
export const EPI_STUFE: Record<string, 1 | 2 | 3 | 4> = {
  code: 1, begriff: 1, referenz: 1,
  begriffsarbeit: 2, aktant: 2,
  konzept: 3, position: 3, relation: 3,
  these: 4, spannung: 4, frage: 4, "ep-ding": 4, "frage-diffraktiv": 4,
};

export function epiClass(epi: EpiKind): string {
  return typeof epi === "number" ? `epi-s${epi}` : `epi-${epi}`;
}

/** Sorten, deren ÜBERSCHRIFT selbst LLM-Formulierung ist — sie folgt dem
    Farb-Code (User-Regel 2026-08-06). Relationen zählen dazu: die Phrasen
    sind nah am Text, aber die VERKNÜPFUNG (X —verb→ Y) ist Modell-Deutung.
    In Tinte bleiben nur echte Wortlaute des Texts: Begriffe, Aktanten,
    Referenzen, Begriffsarbeits-Terme. */
export const EPI_LLM_LABEL = new Set([
  "these", "spannung", "frage", "konzept", "position",
  "ep-ding", "frage-diffraktiv", "code", "relation",
]);

/** Default-Rampe (= :root in styles.css, Rampe B „Blau leuchtet"). */
export const EPI_FARBEN_DEFAULT = ["#0B3D22", "#0A5548", "#0F6494", "#2469F2"];

/** "『#…… #…… #…… #……』"-String → vier valide Hex-Werte (sonst Default). */
export function parseEpiFarben(v: string | null | undefined): string[] {
  const teile = String(v || "").trim().split(/[\s,;]+/).filter(Boolean);
  const hex = teile.filter((t) => /^#[0-9a-fA-F]{6}$/.test(t));
  return hex.length === 4 ? hex : [...EPI_FARBEN_DEFAULT];
}

// ---------- User-Code-Palette (Tool 9, 2026-08-11) ----------
// 6 Markierungs-Farben für Codes; Einstellungen-Key ui_code_farben
// ("#hex ×6", leer = Default — Textmarker-Töne).

export const CODE_FARBEN_DEFAULT = ["#f5d90a", "#4d9df0", "#5bc06c",
                                    "#ef7d54", "#b07cf0", "#4eccc4"];

// EINE Palette für beide Farbräume (User 2026-08-12): der Modus ändert
// nicht die Farbe, nur ob sie über das Papierweiss hinausragt.
export function parseCodeFarben(v: string | null | undefined): string[] {
  const teile = String(v || "").trim().split(/[\s,;]+/).filter(Boolean);
  const hex = teile.filter((t) => /^#[0-9a-fA-F]{6}$/.test(t));
  return hex.length === 6 ? hex : [...CODE_FARBEN_DEFAULT];
}

// HDR-Faktor (User 2026-08-12): Dosis des Neon-Leuchtens, Key
// ui_code_neon_faktor. 1.0 = die kalibrierte Marker-Formel vom 11.08.;
// die Grenzen sind die vom User gesetzten (0.0 = ganz ohne Sättigung …
// 1.5 = grell). 0.0 ist ein GÜLTIGER Wert — darum wird ein fehlender
// Wert an der Leere erkannt, nicht am Nullwert (Number("") ist 0).
export const NEON_FAKTOR_DEFAULT = 1.0;
export const NEON_FAKTOR_MIN = 0.0;
export const NEON_FAKTOR_MAX = 1.5;

export function parseNeonFaktor(v: string | number | null | undefined): number {
  if (v == null || (typeof v === "string" && v.trim() === "")) {
    return NEON_FAKTOR_DEFAULT;
  }
  const n = Number(v);
  if (!Number.isFinite(n)) return NEON_FAKTOR_DEFAULT;
  return Math.min(NEON_FAKTOR_MAX, Math.max(NEON_FAKTOR_MIN, n));
}

/** Wie hoch steigt die Kette `sepia(1) saturate(s) hue-rotate(d)` auf
    weissem Papier? (brightness bleibt draussen — es ist ein Skalar und
    vertauscht mit den Matrizen.) OHNE Klemme gerechnet, weil WebKit im
    HDR-Modus nicht klemmt; der Wert über 1 IST der Überschuss, aus dem
    das Leuchten entsteht. Matrizen nach CSS-Filter-Spezifikation. */
function ketteAufWeiss(s: number, deg: number): [number, number, number] {
  const mul = (m: number[], v: number[]): [number, number, number] => [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
  const sepia = [0.393, 0.769, 0.189,
                 0.349, 0.686, 0.168,
                 0.272, 0.534, 0.131];
  const saturate = [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s,
                    0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s,
                    0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s];
  const a = (deg * Math.PI) / 180, c = Math.cos(a), n = Math.sin(a);
  const hue = [
    0.213 + c * 0.787 - n * 0.213, 0.715 - c * 0.715 - n * 0.715,
    0.072 - c * 0.072 + n * 0.928,
    0.213 - c * 0.213 + n * 0.143, 0.715 + c * 0.285 + n * 0.140,
    0.072 - c * 0.072 - n * 0.283,
    0.213 - c * 0.213 - n * 0.787, 0.715 - c * 0.715 + n * 0.715,
    0.072 + c * 0.928 + n * 0.072,
  ];
  return mul(hue, mul(saturate, mul(sepia, [1, 1, 1])));
}

/** Textmarker-Filter — die Formel vom 11.08. (User: „Das Gelb ist
    schön"), wiederhergestellt am 12.08. nach zwei Fehlversuchen:
    backdrop-filter mit sepia → brightness → saturate → hue-rotate.
    Multiply-SEMANTIK ohne mix-blend-mode (WebKit blendet nicht
    zuverlässig über der GPU-Canvas): das Papier wird Farbe, die Tinte
    bleibt schwarz, weil alle Glieder linear sind.

    NICHT UMBAUEN, ohne es in der APP gesehen zu haben. Zwei Versuche vom
    12.08. sind gescheitert: (1) die Kette je Farbe numerisch „lösen" und
    die Helligkeit ans ENDE stellen — sah in einer Chromium-Render-Probe
    auf 1/255 perfekt aus und war in der App überdrehtes Neon, weil
    CHROMIUM nach jedem Filter-Glied auf [0,1] klemmt und WEBKIT NICHT:
    die Lösung rechnete mit einer Klemme, die es dort nicht gibt.
    (2) Der Faktor nur im HDR-Modus — dann war SDR unregelbar satt.
    Der Überschuss, der auf XDR-Panels leuchtet, entsteht GENAU HIER: aus
    saturate, mit hue-rotate als letztem Glied, das nichts zurückholt
    (Messtafel sandbox/hdr/farbraum-messtafel.html, Felder 6/7/8).

    Hue-abhängige Grunddosis: Gelb voll (5), ferne Töne pastelliger (3).
    `faktor` (Regler 0.0–1.5, User 2026-08-12: „0.2 war top") dosiert sie
    in BEIDEN Modi.

    Der MODUS ändert allein die Helligkeit (User 2026-08-12: „der Schalter
    ist verwaist, er schaltet nicht mehr in den SDR-Modus" — richtig, denn
    macOS gibt einen einmal geöffneten HDR-Modus nicht mehr her; also darf
    nicht das Fenster, sondern muss die MARKIERUNG aufhören zu
    überschiessen):
    · hdr = 0.88 wie gehabt, die Spitze liegt über 1 → Leuchten.
    · sdr = so weit gesenkt, dass die Spitze auf 1 sitzt (bei Faktor 0.2
      etwa 0.73–0.79). Gerechnet wird OHNE Klemme, denn genau so rechnet
      WebKit — das ist dieselbe Einsicht wie oben, nur richtig herum
      angewandt: die Klemme nicht voraussetzen, sondern selbst herstellen.

    Seit 2026-08-15 im Kit (Studio-Dauer-Marker teilen die Optik) —
    UNVERÄNDERT aus UserCodesModule gehoben. */
export function markerFilter(hex: string, hdr: boolean, faktor = 1): string {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return `sepia(1) saturate(${(2 * faktor).toFixed(2)})`;
  const n = parseInt(m[1], 16);
  const r = ((n >> 16) & 255) / 255;
  const g = ((n >> 8) & 255) / 255;
  const bl = (n & 255) / 255;
  const max = Math.max(r, g, bl);
  const min = Math.min(r, g, bl);
  const d = max - min;
  let h = 0;
  if (d > 0) {
    if (max === r) h = ((g - bl) / d) % 6;
    else if (max === g) h = (bl - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const dist = Math.min(Math.abs(h - 55), 360 - Math.abs(h - 55));
  const t = Math.max(0, Math.min(1, (dist - 15) / 35));
  const sat = (5 - 2 * t) * faktor;
  const deg = Math.round(h - 45);
  let hell = 0.88;
  if (!hdr) {
    const spitze = Math.max(...ketteAufWeiss(sat, deg));
    if (spitze > 0 && spitze * hell > 1) hell = 1 / spitze;
  }
  return `sepia(1) brightness(${hell.toFixed(3)}) saturate(${sat.toFixed(2)}) `
    + `hue-rotate(${deg}deg)`;
}

/** Farb-Slot-Wähler ("1".."6" oder null): 6 Swatches + „ohne". */
export function FarbWahl({ value, onChange, farben }: {
  value: string | null; onChange: (v: string | null) => void;
  farben: string[];
}) {
  return (
    <Flex gap="1" align="center">
      {farben.map((f, i) => {
        const key = String(i + 1);
        return (
          <button key={key} type="button" aria-label={`Farbe ${key}`}
                  onClick={() => onChange(value === key ? null : key)}
                  style={{ width: 22, height: 22, borderRadius: "50%",
                           background: f, cursor: "pointer",
                           border: value === key
                             ? "2px solid var(--gray-12)"
                             : "1px solid var(--gray-a6)" }} />
        );
      })}
    </Flex>
  );
}

/** 6 Color-Picker für die User-Code-Palette (Einstellungen). */
export function CodeFarbenInput({ value, onChange }: {
  value: string; onChange: (v: string) => void;
}) {
  const farben = parseCodeFarben(value);
  return (
    <Flex gap="3" align="center" wrap="wrap">
      {farben.map((f, i) => (
        <label key={i} style={{ display: "inline-flex",
                                alignItems: "center", gap: 6,
                                cursor: "pointer" }}>
          <input type="color" value={f} aria-label={`Code-Farbe ${i + 1}`}
                 style={{ width: 34, height: 26, padding: 0,
                          border: "1px solid var(--gray-a6)",
                          borderRadius: 4, background: "none",
                          cursor: "pointer" }}
                 onChange={(e) => {
                   const neu = [...farben];
                   neu[i] = e.target.value;
                   onChange(neu.join(" "));
                 }} />
        </label>
      ))}
      <Button size="1" variant="soft" color="gray"
              onClick={() => onChange("")}>Zurücksetzen</Button>
    </Flex>
  );
}

/** Eigene Farbwerte anwenden (App-Start + Einstellungen, sofort wirksam):
    setzt --epi-s1…s4 als Inline-Properties auf <html>; ungültig/leer ⇒
    zurück auf die styles.css-Defaults. */
export function applyEpiFarben(v: string | null | undefined): void {
  const root = document.documentElement.style;
  const eigen = String(v || "").trim().length > 0;
  const farben = parseEpiFarben(v);
  for (let i = 0; i < 4; i++) {
    if (eigen) root.setProperty(`--epi-s${i + 1}`, farben[i]);
    else root.removeProperty(`--epi-s${i + 1}`);
  }
}

/** Vier Color-Picker für die Nähe-Stufen s1…s4 (Einstellungen). */
export function EpiFarbenInput({ value, onChange }: {
  value: string; onChange: (v: string) => void;
}) {
  const farben = parseEpiFarben(value);
  return (
    <Flex gap="3" align="center" wrap="wrap">
      {farben.map((f, i) => (
        <label key={i} style={{ display: "inline-flex", alignItems: "center",
                                gap: 6, cursor: "pointer" }}>
          <input type="color" value={f}
                 aria-label={`Stufe ${i + 1}`}
                 style={{ width: 34, height: 26, padding: 0, border:
                          "1px solid var(--gray-a6)", borderRadius: 4,
                          background: "none", cursor: "pointer" }}
                 onChange={(e) => {
                   const neu = [...farben];
                   neu[i] = e.target.value;
                   onChange(neu.join(" "));
                 }} />
          <Text size="1" color="gray">s{i + 1}</Text>
        </label>
      ))}
      <Button size="1" variant="ghost"
              onClick={() => onChange(EPI_FARBEN_DEFAULT.join(" "))}>
        Standard</Button>
    </Flex>
  );
}

/** Text mit epistemischer Einfärbung — Radix-Text plus Epistemik-Klasse.
    Trägt immer P1 (User 2026-08-06): ALLE Nicht-Zitat-Texte in derselben
    serifenlosen Schrift und Größe. */
export function EpiText({ epi, className, ...props }:
    { epi: EpiKind } & ComponentProps<typeof Text>) {
  return <Text {...props}
               className={[epiClass(epi), "epi-p1", className]
                 .filter(Boolean).join(" ")} />;
}

export function StatusDot({ tone, pulse }: {
  tone: "ok" | "warn" | "err" | "busy" | "idle";
  /** Blau pulsierend = läuft GERADE (User 2026-08-04, Batch-Live-Anzeige). */
  pulse?: boolean;
}) {
  const color = { ok: "var(--green-9)", warn: "var(--amber-9)", err: "var(--red-9)",
                  busy: "var(--accent-9)", idle: "var(--gray-6)" }[tone];
  return <span className={pulse ? "ui-dot ui-pulse" : "ui-dot"}
               style={{ background: color }} />;
}

/** Ampel-Zeile für Backend-Status (Ollama/Zotero/Web). */
export function HealthDots({ items }: { items: { label: string; tone: "ok" | "err" | "idle" }[] }) {
  return (
    <Flex gap="3" align="center">
      {items.map((s) => (
        <Flex key={s.label} gap="1" align="center">
          <StatusDot tone={s.tone} />
          <Text size="1" color="gray">{s.label}</Text>
        </Flex>
      ))}
    </Flex>
  );
}

// ---------- Jobs (Hub-Ticker + Läufe-Liste teilen sich diese Zeile) ----------

export function jobTone(status: string): "busy" | "ok" | "err" {
  return status === "running" ? "busy" : status === "done" ? "ok" : "err";
}

export function JobRow({ label, status, selected, onClick }: {
  label: string; status: string; selected?: boolean; onClick: () => void;
}) {
  return (
    <ListRow selected={selected} onClick={onClick}
             leading={<StatusDot tone={jobTone(status)} />}
             title={label}
             trailing={<Badge color={status === "running" ? "blue" : status === "done" ? "green" : "red"}>
               {status === "running" ? "läuft" : status === "done" ? "fertig" : "Fehler"}
             </Badge>} />
  );
}

// ---------- Tabelle (Kit-Baustein, User 2026-08-02) ----------
// EINE sortier-/scrollbare Tabelle für Import-Bibliothek, Dossier-
// Verwaltung u. a. — Filter/Suche macht der Aufrufer (er reicht die
// gefilterten rows), die Tabelle liefert Sortierung, Sticky-Kopf und
// optional die Auswahl-Spalte.

export type ColumnDef<T> = {
  key: string;
  label: ReactNode;
  width?: number | string;
  /** vorhanden = Spalte sortierbar (Klick auf den Kopf; null ans Ende) */
  sortValue?: (row: T) => string | number | null;
  render: (row: T) => ReactNode;
};

export function DataTable<T>({ columns, rows, rowKey, selected, onToggle,
                               onToggleAll, selectable, defaultSort, empty,
                               cap }: {
  columns: ColumnDef<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  /** Auswahl-Spalte erscheint, wenn selected + onToggle gesetzt sind. */
  selected?: Set<string>;
  onToggle?: (key: string, checked: boolean) => void;
  /** Kopf-Checkbox: alle SICHTBAREN (gefilterten) Zeilen an/abwählen. */
  onToggleAll?: (keys: string[], checked: boolean) => void;
  /** true = Zeile wählbar; ein String = NICHT wählbar, der String ist der
      Grund (Tooltip an der gesperrten Checkbox; Kopf-Checkbox lässt die
      Zeile aus). */
  selectable?: (row: T) => true | string;
  defaultSort?: { key: string; dir: "asc" | "desc" };
  empty?: ReactNode;
  /** Render-Deckel NACH der Sortierung — abgeschnittene Zeilen werden als
      ehrliche „… X weitere"-Zeile ausgewiesen (kein silent cap). */
  cap?: number;
}) {
  const [sort, setSort] = useState(defaultSort ?? null);
  const sorted = useMemo(() => {
    const col = sort ? columns.find((c) => c.key === sort.key) : undefined;
    if (!sort || !col?.sortValue) return rows;
    const mul = sort.dir === "asc" ? 1 : -1;
    const val = col.sortValue;
    return [...rows].sort((a, b) => {
      const va = val(a), vb = val(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;  // Leeres immer ans Ende, egal welche Richtung
      if (vb == null) return -1;
      return (typeof va === "number" && typeof vb === "number"
        ? va - vb : String(va).localeCompare(String(vb), "de")) * mul;
    });
  }, [rows, sort, columns]);
  const shown = cap != null && sorted.length > cap ? sorted.slice(0, cap) : sorted;
  const cut = sorted.length - shown.length;
  const keys = useMemo(
    () => shown.filter((r) => (selectable?.(r) ?? true) === true).map(rowKey),
    [shown, rowKey, selectable]);
  const allChecked = !!selected && keys.length > 0
    && keys.every((k) => selected.has(k));
  const withSelect = !!selected && !!onToggle;

  const head = (c: ColumnDef<T>) => {
    if (!c.sortValue) return c.label;
    const active = sort?.key === c.key;
    return (
      <button type="button" className="ui-th-sort"
              aria-sort={active ? (sort.dir === "asc" ? "ascending" : "descending") : undefined}
              onClick={() => setSort((s) => s?.key === c.key
                ? { key: c.key, dir: s.dir === "asc" ? "desc" : "asc" }
                : { key: c.key, dir: "asc" })}>
        {c.label}{active && (sort.dir === "asc" ? " ▲" : " ▼")}
      </button>
    );
  };

  return (
    <Table.Root size="1" className="ui-datatable">
      <Table.Header>
        <Table.Row>
          {withSelect && (
            <Table.ColumnHeaderCell style={{ width: 32 }}>
              <Checkbox size="1" checked={allChecked}
                        aria-label="Alle sichtbaren an/abwählen"
                        onCheckedChange={(c) => onToggleAll?.(keys, c === true)} />
            </Table.ColumnHeaderCell>
          )}
          {columns.map((c) => (
            <Table.ColumnHeaderCell key={c.key} style={c.width ? { width: c.width } : undefined}>
              {head(c)}
            </Table.ColumnHeaderCell>
          ))}
        </Table.Row>
      </Table.Header>
      <Table.Body>
        {sorted.length === 0 && (
          <Table.Row>
            <Table.Cell colSpan={columns.length + (withSelect ? 1 : 0)}>
              <Text size="1" color="gray">{empty ?? "Keine Einträge."}</Text>
            </Table.Cell>
          </Table.Row>
        )}
        {shown.map((row) => {
          const k = rowKey(row);
          const sel = selectable?.(row) ?? true;
          return (
            <Table.Row key={k} className={selected?.has(k) ? "selected" : undefined}>
              {withSelect && (
                <Table.Cell>
                  {sel === true ? (
                    <Checkbox size="1" checked={selected.has(k)}
                              aria-label="Zeile auswählen"
                              onCheckedChange={(c) => onToggle?.(k, c === true)} />
                  ) : (
                    // disabled feuert keine Tooltips — der title sitzt am Span
                    <span title={sel}>
                      <Checkbox size="1" checked={false} disabled
                                aria-label={sel} />
                    </span>
                  )}
                </Table.Cell>
              )}
              {columns.map((c) => (
                <Table.Cell key={c.key}>{c.render(row)}</Table.Cell>
              ))}
            </Table.Row>
          );
        })}
        {cut > 0 && (
          <Table.Row>
            <Table.Cell colSpan={columns.length + (withSelect ? 1 : 0)}>
              <Text size="1" color="gray">
                … {cut} weitere — Suche oder Filter eingrenzen.
              </Text>
            </Table.Cell>
          </Table.Row>
        )}
      </Table.Body>
    </Table.Root>
  );
}

// ---------- Formulare ----------

export function Field({ label, hint, origin, value, onChange, placeholder,
                        children }: {
  label: string; hint?: string; origin?: string; placeholder?: string;
  value: string; onChange: (v: string) => void;
  /** Optionales eigenes Eingabe-Element (z. B. PathPicker) — ersetzt das
      Standard-Textfeld, behält Label/Hinweis/Herkunft. */
  children?: ReactNode;
}) {
  const id = useId();  // Label programmatisch verdrahtet (Prüfbericht #35)
  return (
    <Box mb="3">
      <Text as="label" size="1" color="gray" htmlFor={id}>
        {label}{hint ? ` — ${hint}` : ""}
        {origin && <Text size="1" color="blue"> · {origin}</Text>}
      </Text>
      {children ?? (
        <TextField.Root id={id} mt="1" value={value} placeholder={placeholder}
                        onChange={(e) => onChange(e.target.value)} />
      )}
    </Box>
  );
}

/** Zahlen-Regler mit Wert-Anzeige (Kit-Baustein 2026-08-12, HDR-Faktor):
    für stufenlose Dosierungen, wo eine Zahl im Textfeld nichts sagt und
    das AUGE entscheidet. Der Wert steht rechts mit, damit er zitierbar
    bleibt (Reproduzierbarkeit einer Einstellung). */
export function Regler({ label, hint, origin, value, min, max, step = 0.05,
                          format, onChange }: {
  label: string; hint?: string; origin?: string;
  value: number; min: number; max: number; step?: number;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}) {
  const zeig = format ?? ((v: number) => v.toFixed(2));
  return (
    <Box mb="3">
      <Flex align="baseline" justify="between" gap="2">
        <Text as="div" size="1" color="gray">
          {label}{hint ? ` — ${hint}` : ""}
          {origin && <Text size="1" color="blue"> · {origin}</Text>}
        </Text>
        <Text size="1" style={{ fontVariantNumeric: "tabular-nums" }}>
          {zeig(value)}
        </Text>
      </Flex>
      <Box mt="2">
        <Slider size="1" min={min} max={max} step={step} value={[value]}
                onValueChange={(v) => onChange(v[0] ?? value)} />
      </Box>
    </Box>
  );
}

export function SearchField({ value, onChange, placeholder }: {
  value: string; onChange: (v: string) => void; placeholder: string;
}) {
  return (
    <TextField.Root type="search" value={value} placeholder={placeholder}
                    onChange={(e) => onChange(e.target.value)} />
  );
}

/** Segment-Umschalter (Quelle wählen, Geltungsbereich …). */
export function SegTabs({ value, onChange, options }: {
  value: string; onChange: (v: string) => void;
  // icon optional (2026-08-07, Textmengen-Switch im Fragen-Tab): Icon
  // und/oder Label; title trägt die Erklärung (Tooltip + a11y).
  options: { value: string; label: string; icon?: IconName;
             title?: string }[];
}) {
  return (
    <SegmentedControl.Root value={value} onValueChange={onChange} size="1">
      {options.map((o) => (
        <SegmentedControl.Item key={o.value} value={o.value}
                               title={o.title} aria-label={o.title || o.label}>
          {o.icon && <Icon name={o.icon} size={13} />}
          {o.icon && o.label ? " " : ""}{o.label}
        </SegmentedControl.Item>
      ))}
    </SegmentedControl.Root>
  );
}

/** Sterne-Bewertung 1–5 (Chat-Feedback, User 2026-08-08): ein Klick
    bewertet, erneutes Klicken ändert; tastaturfähig. */
export function Sterne({ value, onChange, title }: {
  value: number | null | undefined;
  onChange: (n: number) => void; title?: string;
}) {
  return (
    <span title={title} style={{ whiteSpace: "nowrap" }}>
      {[1, 2, 3, 4, 5].map((n) => (
        <span key={n} role="button" tabIndex={0}
              aria-label={`${n} Sterne`}
              onClick={() => onChange(n)}
              onKeyDown={(e) => e.key === "Enter" && onChange(n)}
              style={{ cursor: "pointer", fontSize: 14, padding: "0 1px",
                       color: (value ?? 0) >= n
                         ? "var(--amber-9)" : "var(--gray-6)" }}>
          ★
        </span>
      ))}
    </span>
  );
}


/** Schwebe-Karte (User 2026-08-08): Link-Preview beim Hover — für
    Chat-Zitationen (Ast/Wortlaut), generisch nutzbar. Klick auf den
    Trigger bleibt frei (z. B. Studio-Sprung). */
export function SchwebeKarte({ trigger, children, onOpenChange }: {
  trigger: ReactNode; children: ReactNode;
  onOpenChange?: (open: boolean) => void;
}) {
  return (
    <HoverCard.Root openDelay={250} closeDelay={150}
                    onOpenChange={onOpenChange}>
      <HoverCard.Trigger>{trigger}</HoverCard.Trigger>
      <HoverCard.Content size="1"
                         style={{ maxWidth: 480, maxHeight: 400,
                                  overflowY: "auto" }}>
        {children}
      </HoverCard.Content>
    </HoverCard.Root>
  );
}


/** Klickbarer Filter-Chip: Legende, die zugleich filtert (Graph-Ebenen).
    Tastaturfähig (echter Button); inaktiv = ausgegraut. */
export function FilterChip({ active, color, onClick, children }: {
  active: boolean; color: ComponentProps<typeof Badge>["color"];
  onClick: () => void; children: ReactNode;
}) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active}
            style={{ all: "unset", cursor: "pointer",
                     opacity: active ? 1 : 0.35 }}>
      <Badge color={color} variant={active ? "soft" : "outline"}>{children}</Badge>
    </button>
  );
}

/** Typ-Pulldown DIREKT an einer BBox (Viewer-Overlays, Prüfbericht #28):
    farbiges Label als Trigger, Optionen als Radix-Select. */
export function BBoxTypeSelect({ value, color, options, onChange }: {
  value: string; color: string; options: readonly string[];
  onChange: (v: string) => void;
}) {
  return (
    <Select.Root size="1" value={value} onValueChange={onChange}>
      {/* Farbe als CSS-Variable: der HINTERGRUND wird im Stylesheet
          halbtransparent gemischt (50 %/70 % hover), die weiße Schrift
          bleibt 100 % deckend (User 2026-08-13). */}
      <Select.Trigger className="ui-bboxlabel"
                      style={{ "--bbox-farbe": color } as CSSProperties} />
      <Select.Content>
        {options.map((o) => (
          <Select.Item key={o} value={o}>{o}</Select.Item>
        ))}
      </Select.Content>
    </Select.Root>
  );
}

/** Beschriftete Auswahl, z. B. Ziel-Projekt. */
export function LabeledSelect({ label, value, onChange, options, emptyLabel,
                                optionLabels }: {
  label: string; value: string; onChange: (v: string) => void;
  options: string[]; emptyLabel?: string;
  /** Anzeige-Text je Options-Wert (Wert bleibt der Schlüssel). */
  optionLabels?: Record<string, string>;
}) {
  const EMPTY = " none";
  return (
    <Flex align="center" gap="2">
      <Text size="1" color="gray">{label}</Text>
      <Select.Root value={value || EMPTY}
                   onValueChange={(v) => onChange(v === EMPTY ? "" : v)} size="1">
        <Select.Trigger aria-label={label || "Auswahl"} />
        <Select.Content>
          {emptyLabel !== undefined && <Select.Item value={EMPTY}>{emptyLabel}</Select.Item>}
          {options.map((o) => <Select.Item key={o} value={o}>{optionLabels?.[o] ?? o}</Select.Item>)}
        </Select.Content>
      </Select.Root>
    </Flex>
  );
}

// ---------- Hub-Kacheln + TabBar ----------

export function Tile({ icon, title, meta, disabled, onClick, drag }: {
  icon: IconName; title: string; meta?: string; disabled?: boolean; onClick: () => void;
  /** Sortieren per Ziehen (Hub 2026-08-02): der Aufrufer hält die Ordnung,
      die Kachel liefert nur Griff + Ablage-Optik. */
  drag?: {
    dragging: boolean; dropTarget: boolean;
    onStart: () => void; onEnter: () => void; onDrop: () => void; onEnd: () => void;
  };
}) {
  return (
    <button className={"ui-tile" + (disabled ? " planned" : "")
                       + (drag?.dragging ? " dragging" : "")
                       + (drag?.dropTarget ? " dragover" : "")}
            disabled={disabled} onClick={onClick}
            draggable={!!drag && !disabled}
            onDragStart={drag ? (e) => {
              // Firefox startet den Drag nur mit gesetztem dataTransfer
              e.dataTransfer.setData("text/plain", "");
              e.dataTransfer.effectAllowed = "move";
              drag.onStart();
            } : undefined}
            onDragEnter={drag ? () => drag.onEnter() : undefined}
            onDragOver={drag ? (e) => e.preventDefault() : undefined}
            onDrop={drag ? (e) => { e.preventDefault(); drag.onDrop(); } : undefined}
            onDragEnd={drag ? () => drag.onEnd() : undefined}>
      <span className="ui-tile-icon"><Icon name={icon} size={20} /></span>
      <Text size="2" weight="medium">{title}</Text>
      {meta && <Text size="1" color="gray">{meta}</Text>}
    </button>
  );
}

export type TabDef = {
  id: string; title: string; icon?: IconName; closable?: boolean;
  /** Fester Anker (z. B. „Hub"): nicht zieh- und nicht verdrängbar. */
  fixed?: boolean;
  /** Läuft als eigenes Fenster — gedimmt dargestellt. */
  detached?: boolean;
};

export function TabBar({ tabs, active, onSelect, onClose, onReorder }: {
  tabs: TabDef[]; active: string;
  onSelect: (id: string) => void; onClose?: (id: string) => void;
  /** Sortieren per Ziehen (User 2026-08-02): src wird VOR dst eingefügt;
      der Aufrufer hält und persistiert die Ordnung. */
  onReorder?: (src: string, dst: string) => void;
}) {
  // Quelle des Zugs im Ref (Drop liest SYNCHRON — State wäre beim
  // schnellen Drop noch nicht committed); State nur für die Optik.
  const dragRef = useRef<string | null>(null);
  const [dragId, setDragId] = useState<string | null>(null);
  const [overId, setOverId] = useState<string | null>(null);
  return (
    <div className="ui-tabbar" role="tablist">
      {tabs.map((t) => {
        const draggable = !!onReorder && !t.fixed;
        return (
          <div key={t.id} role="tab" aria-selected={t.id === active}
               tabIndex={0}  // Tastatur: Enter/Space wählt (Prüfbericht #31)
               className={"ui-tab" + (t.id === active ? " active" : "")
                          + (t.detached ? " detached" : "")
                          + (dragId === t.id ? " dragging" : "")
                          + (overId === t.id && dragId !== null && dragId !== t.id
                             ? " dragover" : "")}
               title={t.detached ? "läuft als eigenes Fenster" : undefined}
               onClick={() => onSelect(t.id)}
               onKeyDown={(e) => {
                 if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(t.id); }
               }}
               draggable={draggable}
               onDragStart={draggable ? (e) => {
                 // Firefox startet den Drag nur mit gesetztem dataTransfer
                 e.dataTransfer.setData("text/plain", "");
                 e.dataTransfer.effectAllowed = "move";
                 dragRef.current = t.id; setDragId(t.id);
               } : undefined}
               onDragEnter={draggable ? () => setOverId(t.id) : undefined}
               onDragOver={draggable ? (e) => e.preventDefault() : undefined}
               onDrop={draggable ? (e) => {
                 e.preventDefault();
                 const src = dragRef.current;
                 if (src && src !== t.id) onReorder(src, t.id);
                 dragRef.current = null; setDragId(null); setOverId(null);
               } : undefined}
               onDragEnd={draggable ? () => {
                 dragRef.current = null; setDragId(null); setOverId(null);
               } : undefined}>
            {t.icon && <span className="ui-tab-icon"><Icon name={t.icon} size={13} /></span>}
            <span>{t.title}</span>
            {t.closable && onClose && (
              <button type="button" className="ui-tab-close" title="Tab schließen"
                      aria-label={`${t.title} schließen`}
                      onClick={(e) => { e.stopPropagation(); onClose(t.id); }}>×</button>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function IconButton({ title, onClick, children }: {
  title: string; onClick: () => void; children: ReactNode;
}) {
  return (
    <Button variant="ghost" color="gray" size="1" title={title} onClick={onClick}
            style={{ fontSize: 15 }}>
      {children}
    </Button>
  );
}

/** Merkt die Scroll-Position des nächsten scrollenden VORFAHREN je
    storeKey (sessionStorage) und stellt sie beim Mount wieder her —
    für Ansichts-Wechsel in Seitenpanels (Studio: Baum ⇄ Liste ⇄ Index).
    Wächst der Inhalt nach dem Mount noch (asynchron geladene Zeilen),
    wird die Zielposition nachgezogen, solange der User nicht selbst
    gescrollt hat. */
export function ScrollKeeper({ storeKey, children }: {
  storeKey: string; children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  useLayoutEffect(() => {
    let el: HTMLElement | null = ref.current?.parentElement ?? null;
    while (el) {
      const o = getComputedStyle(el).overflowY;
      if (o === "auto" || o === "scroll") break;
      el = el.parentElement;
    }
    if (!el) return;
    const box = el;
    const want = Number(sget(storeKey)) || 0;
    box.scrollTop = want;
    let applied = box.scrollTop;
    let raf = 0;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => sset(storeKey, String(box.scrollTop)));
    };
    box.addEventListener("scroll", onScroll, { passive: true });
    const ro = new ResizeObserver(() => {
      if (Math.abs(box.scrollTop - applied) < 2 && box.scrollTop < want) {
        box.scrollTop = want;
        applied = box.scrollTop;
      }
    });
    if (ref.current) ro.observe(ref.current);
    return () => {
      cancelAnimationFrame(raf);
      box.removeEventListener("scroll", onScroll);
      ro.disconnect();
    };
  }, [storeKey]);
  return <div ref={ref}>{children}</div>;
}

// ---------- SidePanel: einklappbare Seitenleiste ----------
// UX-Muster als Komponente (User-Vorgabe): minimiert zu einem schmalen
// Streifen mit Knopf; optional per Handle in der Breite ziehbar (bis
// maxFraction der Fensterbreite). Zustand pro Fenster in sessionStorage.

export function SidePanel({ side, title, storageKey, defaultWidth = 300,
  minWidth = 240, maxFraction = 0.5, resizable = false, headerExtra,
  headTone, children }: {
  side: "left" | "right";
  title: ReactNode;
  storageKey: string;
  defaultWidth?: number;
  minWidth?: number;
  maxFraction?: number;
  resizable?: boolean;
  headerExtra?: ReactNode;
  /** "accent" = der Kopf spiegelt eine aktive Auswahl (Codebuch-
      Editor 2026-08-28) — gleiche Tönung wie gewählte Zeilen. */
  headTone?: "accent";
  children: ReactNode;
}) {
  const [open, setOpen] = useState(() => sget(`${storageKey}.open`) !== "0");
  const [width, setWidth] = useState(() =>
    Number(sget(`${storageKey}.w`)) || defaultWidth);
  const widthRef = useRef(width);
  widthRef.current = width;
  const drag = useRef<{ x: number; w: number } | null>(null);

  useEffect(() => {
    if (!resizable) return;
    const onMove = (e: MouseEvent) => {
      const st = drag.current;
      if (!st) return;
      const delta = side === "right" ? st.x - e.clientX : e.clientX - st.x;
      const max = Math.round(window.innerWidth * maxFraction);
      setWidth(Math.min(max, Math.max(minWidth, st.w + delta)));
    };
    const onUp = () => {
      if (drag.current) sset(`${storageKey}.w`, String(widthRef.current));
      drag.current = null;
      document.body.style.cursor = "";
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [resizable, side, minWidth, maxFraction, storageKey]);

  const toggle = () => setOpen((cur) => {
    sset(`${storageKey}.open`, cur ? "0" : "1");
    return !cur;
  });

  if (!open) {
    return (
      <div className={`ui-sidestrip ${side}`}>
        <IconButton title="Einblenden" onClick={toggle}>
          {side === "left" ? "⟩" : "⟨"}
        </IconButton>
      </div>
    );
  }
  const handle = resizable ? (
    <div className="ui-drag-handle"
         onMouseDown={(e) => {
           drag.current = { x: e.clientX, w: widthRef.current };
           document.body.style.cursor = "col-resize";
           e.preventDefault();
         }} />
  ) : null;
  const panel = (
    <Flex direction="column" className={`ui-sidepanel ${side}`}
          style={{ width: resizable ? width : defaultWidth, flexShrink: 0 }}>
      <Flex px="3" py="2" gap="2" align="center" className="ui-sidehead"
            style={headTone === "accent"
              ? { background: "var(--accent-a3)" } : undefined}>
        {side === "right" && <IconButton title="Ausblenden" onClick={toggle}>⟩</IconButton>}
        <Text size="1" weight="medium" color="gray"
              style={{ flex: 1, minWidth: 0 }} truncate>{title}</Text>
        {headerExtra}
        {side === "left" && <IconButton title="Ausblenden" onClick={toggle}>⟨</IconButton>}
      </Flex>
      <Box style={{ flex: 1, minHeight: 0, overflow: "auto" }}>{children}</Box>
    </Flex>
  );
  // Handle sitzt zur CONTENT-Seite hin: rechts-Panel → davor,
  // links-Panel → dahinter (vorher fiel er links still weg).
  return side === "right" ? <>{handle}{panel}</>
                          : <>{panel}{handle}</>;
}

// ---------- DocMap: abstrakte Dokument-Karte (Multi-Color) ----------
// EIN Bauteil für Studio-TextMap UND Diffraktions-Typ-Karte (User
// 2026-07-30): dunkler Streifen, Textblock-SILHOUETTEN als Grundgerüst,
// darüber farbige Markierungen, Seitenlinien und Viewport-Rahmen.
// Marks sind OPAK (User-Vorgabe 2026-07-30): Transparenz über den
// Silhouetten verfälscht die Farben — opacity nur auf explizite Angabe.
// Achsen-agnostisch: alle Positionen kommen als Brüche 0..1 — der
// Aufrufer rechnet seine Achse (Seitenhöhen ODER Text-Offsets) selbst.

export type DocMapBand = {
  key: string;
  top: number;            // 0..1
  height: number;         // 0..1
  left?: number;          // 0..1 (Default 0)
  width?: number;         // 0..1 (Default 1)
  color?: string;         // Markierungsfarbe (Marks); Silhouetten ignorieren
  opacity?: number;
  glow?: boolean;
  active?: boolean;
  title?: string;
  onClick?: () => void;
};

/** Viewport-Rahmen mit EIGENEM Scroll-Zustand (Perf, User 2026-08-15):
    beim Scrollen rendert nur dieses eine div neu — nie die tausenden
    Karten-Elemente daneben (vorher difften 60×/s ~8k Knoten). `signal`
    re-attacht den Listener, wenn der Scroller erst später existiert
    (PDF lädt lazy — pageCount als Signal). */
function DocMapViewport({ scrollRef, signal }: {
  scrollRef: RefObject<HTMLDivElement | null>; signal?: unknown;
}) {
  const [view, setView] =
    useState<{ top: number; height: number } | null>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const update = () => setView(el.scrollHeight > 0
      ? { top: el.scrollTop / el.scrollHeight,
          height: el.clientHeight / el.scrollHeight }
      : null);
    update();
    el.addEventListener("scroll", update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => { el.removeEventListener("scroll", update); ro.disconnect(); };
  }, [scrollRef, signal]);
  if (!view || view.height >= 0.999) return null;
  return (
    <div className="ui-textmap-view"
         style={{ top: `${view.top * 100}%`,
                  height: `${Math.max(1.5, view.height * 100)}%` }} />
  );
}

export function DocMap({ silhouettes, marks, sections, pageLines, viewport,
                         scrollRef, viewportSignal, title, onJump }: {
  silhouettes?: DocMapBand[];   // gedämpfte Textblöcke (Grundgerüst)
  marks?: DocMapBand[];         // farbige Markierungen (Multi-Color)
  //: Sektions-Spur (Gliederung 2026-08-14): halbtransparente Bänder in
  //: voller Breite UNTER Silhouetten und Marks — Kapitel-Chronologie.
  sections?: DocMapBand[];
  pageLines?: number[];         // Seitengrenzen als Brüche 0..1
  viewport?: { top: number; height: number } | null;  // sichtbarer Bereich
  //: Bevorzugt gegenüber `viewport`: der Rahmen verwaltet seinen
  //: Scroll-Zustand SELBST (nur er rendert beim Scrollen neu).
  scrollRef?: RefObject<HTMLDivElement | null>;
  viewportSignal?: unknown;
  title?: string;
  onJump?: (frac: number) => void;  // Klick auf Leerfläche (0..1)
}) {
  // CANVAS für die Massen-Ebenen (Karten-Robustheit, User 2026-08-15):
  // Silhouetten, Marks und Seitenlinien sind bei Monografien tausende
  // Elemente — als DOM diffte React sie bei jeder Änderung komplett.
  // Jetzt EIN Repaint (<2 ms für 10k Rechtecke). DOM bleiben nur die
  // wenigen interaktiven/animierten Teile: Sektions-Bänder (active-
  // Kontur), Glow-Marks (CSS-Animation) und der Viewport-Rahmen.
  // Klick + Hover laufen über rechnerisches Hit-Testing am Container.
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const plainMarks = (marks ?? []).filter((b) => !b.glow);
  const glowMarks = (marks ?? []).filter((b) => b.glow);
  //: Hit-Test in STUFEN: Marks → Sektionen → Silhouetten. Sektionen vor
  //: Silhouetten, sonst gewinnt an fast jedem Punkt eine (kleinere)
  //: Silhouette und der Kapitel-Klick wäre unerreichbar; innerhalb
  //: einer Stufe gewinnt das kleinste enthaltende Band.
  const hitRef = useRef<DocMapBand[][]>([]);
  hitRef.current = [marks ?? [], sections ?? [], silhouettes ?? []];

  useEffect(() => {
    const wrap = wrapRef.current;
    const cv = canvasRef.current;
    if (!wrap || !cv) return;
    const paint = () => {
      const w = wrap.clientWidth, h = wrap.clientHeight;
      if (!w || !h) return;
      const dpr = window.devicePixelRatio || 1;
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      const ctx = cv.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      // Farben = die bisherigen CSS-Werte (der Streifen ist bewusst
      // dunkel in beiden Themes — styles.css .ui-textmap-*)
      ctx.strokeStyle = "#2c2c40";
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      for (const f of pageLines ?? []) {
        const y = Math.round(f * h) + 0.5;
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      // Silhouetten als EIN Pfad mit EINEM Fill (User 2026-08-15, „zu
      // hell"): einzelne fillRects akkumulierten an Überlappungen die
      // Transparenz — ein Pfad füllt jede Fläche genau einmal; dazu
      // dunkler als die alte DOM-Optik (0.15 statt 0.26).
      const silPfad = new Path2D();
      for (const b of silhouettes ?? []) {
        silPfad.rect((b.left ?? 0) * w, b.top * h,
                     Math.max(1, (b.width ?? 1) * w),
                     Math.max(1, b.height * h));
      }
      ctx.fillStyle = "rgba(255,255,255,0.15)";
      ctx.fill(silPfad);
      for (const b of plainMarks) {
        ctx.globalAlpha = b.opacity ?? 1;
        ctx.fillStyle = b.color ?? "#ff5a6e";
        ctx.fillRect((b.left ?? 0.08) * w, b.top * h,
                     Math.max(2, (b.width ?? 0.84) * w),
                     Math.max(2, b.height * h));
        ctx.globalAlpha = 1;
        if (b.active) {
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1.5;
          ctx.strokeRect((b.left ?? 0.08) * w, b.top * h,
                         Math.max(2, (b.width ?? 0.84) * w),
                         Math.max(2, b.height * h));
        }
      }
    };
    paint();
    const ro = new ResizeObserver(paint);
    ro.observe(wrap);
    return () => ro.disconnect();
    // plainMarks ist aus marks abgeleitet — marks als Dep genügt.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [silhouettes, marks, pageLines]);

  const trefferAn = (fx: number, fy: number): DocMapBand | null => {
    for (const stufe of hitRef.current) {
      let best: DocMapBand | null = null;
      for (const b of stufe) {
        const l = b.left ?? 0, wgt = b.width ?? 1;
        if (fy >= b.top && fy <= b.top + Math.max(0.002, b.height)
            && fx >= l && fx <= l + wgt) {
          if (!best || b.height < best.height) best = b;
        }
      }
      if (best) return best;
    }
    return null;
  };
  const hoverRaf = useRef(0);

  return (
    <div className="ui-textmap" title={title} ref={wrapRef}
         onClick={(e) => {
           const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
           const fx = (e.clientX - r.left) / Math.max(1, r.width);
           const fy = (e.clientY - r.top) / Math.max(1, r.height);
           const hit = trefferAn(fx, fy);
           if (hit?.onClick) hit.onClick();
           else onJump?.(fy);
         }}
         onMouseMove={(e) => {
           if (hoverRaf.current) return;
           const el = e.currentTarget as HTMLElement;
           const cx = e.clientX, cy = e.clientY;
           hoverRaf.current = requestAnimationFrame(() => {
             hoverRaf.current = 0;
             const r = el.getBoundingClientRect();
             const hit = trefferAn((cx - r.left) / Math.max(1, r.width),
                                   (cy - r.top) / Math.max(1, r.height));
             const t = hit?.title ?? title ?? "";
             if (el.title !== t) el.title = t;
           });
         }}>
      {(sections ?? []).map((b) => (
        <div key={b.key}
             className={"ui-textmap-sect" + (b.active ? " active" : "")}
             style={{ top: `${b.top * 100}%`,
                      height: `max(2px, ${b.height * 100}%)`,
                      pointerEvents: "none",
                      ...(b.color ? { background: b.color } : {}),
                      ...(b.opacity != null ? { opacity: b.opacity } : {}) }} />
      ))}
      <canvas ref={canvasRef}
              style={{ position: "absolute", inset: 0, width: "100%",
                       height: "100%", pointerEvents: "none" }} />
      {glowMarks.map((b) => (
        <div key={b.key}
             className={"ui-textmap-mark glow"
                        + (b.active ? " active" : "")}
             style={{ top: `${b.top * 100}%`,
                      height: `max(2px, ${b.height * 100}%)`,
                      left: `${(b.left ?? 0.08) * 100}%`,
                      width: `max(2px, ${(b.width ?? 0.84) * 100}%)`,
                      pointerEvents: "none",
                      // Fokus MIT eigener Farbe (User Codes 2026-08-17):
                      // Code-Farbe statt Rosa, OHNE Glow — der rosa
                      // Glow bleibt die Studio-Semantik ohne color.
                      ...(b.color ? { background: b.color,
                                      boxShadow: "none" } : {}) }} />
      ))}
      {scrollRef
        ? <DocMapViewport scrollRef={scrollRef} signal={viewportSignal} />
        : viewport && viewport.height < 0.999 && (
          <div className="ui-textmap-view"
               style={{ top: `${viewport.top * 100}%`,
                        height: `${Math.max(1.5, viewport.height * 100)}%` }} />
        )}
    </div>
  );
}

/** Unbegrenzte, deterministische Farbskala für DocMap-Marks (User
    2026-07-30: Anzahl der Farben unbestimmt): Goldener-Winkel-Rotation
    im HSL-Raum — beliebig viele unterscheidbare Farben, stabil über
    die sortierten Schlüssel. */
export function mapPalette(keys: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  [...keys].sort().forEach((k, i) => {
    const hue = Math.round((i * 137.508) % 360);
    out[k] = `hsl(${hue} 72% 55%)`;
  });
  return out;
}

/** Farb-PAAR je Schlüssel (User 2026-07-30: Pole der Typologie-Paare
    farbig trennen): gleiche Goldener-Winkel-Grundfarbe je Typ, Pol A
    kräftig / Pol B hell — als Paar erkennbar, als Seiten trennbar. */
export function mapPalettePair(keys: string[]
                               ): Record<string, { a: string; b: string }> {
  const out: Record<string, { a: string; b: string }> = {};
  [...keys].sort().forEach((k, i) => {
    const hue = Math.round((i * 137.508) % 360);
    out[k] = { a: `hsl(${hue} 75% 52%)`, b: `hsl(${hue} 65% 78%)` };
  });
  return out;
}
