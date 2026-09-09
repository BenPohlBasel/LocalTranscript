// Editor (Ebene 2, Drilldown): Modell im STATE, nie im DOM (v1-Lehre).
// Autosave (debounced) + History im Backend; Sprecher sind Entitäten
// (umbenennen überall, Segment umhängen, zusammenführen); Segment-Ops
// teilen/verbinden/löschen; Audio-Player mit Folgen-Modus.
import { memo, useCallback, useEffect, useMemo, useRef, useState }
  from "react";
import {
  Badge, Button, Checkbox, ErrorNote, Flex, IconButton,
  SearchField, SegTabs, Select, SidePanel, Text, TextField,
} from "../components/ui";
import { Icon } from "../components/icons";
import {
  API_BASE, apiGet, apiSend, errMsg, hms, kuerze, sprecherFarbe,
  type Segment, type Sprecher, type Transkript,
} from "../lib/api";
import { useT } from "../lib/i18n";
import { KEYS, lget, lset, sget, sset } from "../lib/storage";
import { isTauri, savePath } from "../lib/tauri";

const SPEEDS = [1, 1.25, 1.5, 1.75, 2];

// EIN Zeilen-Slot für alle Zellen einer Segmentzeile (User
// 2026-09-09: „ausrichten der Zeilen"): 28 px = Rahmen 1 + Polster 3
// + Textzeile 19,5 + Polster 3 + Rahmen 1 der Textarea. Alles darin
// zentriert → Play, Timecode, Sprecher-Badge, Text und Aktionen
// sitzen auf derselben Mittellinie (vorher 13,5–18,75 px Streuung).
const SEG_SLOT = { height: 28, display: "flex",
                   alignItems: "center" } as const;

let _seq = 0;
function neueId(): string {
  _seq += 1;
  return `n${Date.now().toString(36)}${_seq}`;
}

export default function EditorModule({ id, onExit }: {
  id: string; onExit: () => void;
}) {
  const tr = useT();
  const [name, setName] = useState("");
  const [sprecher, setSprecher] = useState<Sprecher[]>([]);
  const [segmente, setSegmente] = useState<Segment[]>([]);
  const [hatAudio, setHatAudio] = useState(false);
  const [fehler, setFehler] = useState("");
  const [speichert, setSpeichert] = useState(false);
  const [gespeichert, setGespeichert] = useState("");
  const [aktiv, setAktiv] = useState(-1);
  const [laeuft, setLaeuft] = useState(false);
  const [speed, setSpeed] = useState(
    Number(lget(KEYS.editorSpeed)) || 1);
  const [loop, setLoop] = useState(false);
  const [folgen, setFolgen] = useState(
    lget(KEYS.editorFolgen) !== "0");
  const [seitenTab, setSeitenTab] = useState<"sprecher" | "suchen">(
    sget(KEYS.editorSeitenTab) === "suchen" ? "suchen" : "sprecher");
  // Zähler je Segment: die Textareas sind UNKONTROLLIERT
  // (defaultValue) und zeigen programmatisch geänderten Text nur
  // nach Remount. Statt — wie bei Teilen/Verbinden — die id zu
  // wechseln (die ist kanonische Identität in transkript.json),
  // hängt der Zähler am React-KEY: Ersetzen remountet die Zeile,
  // die Segment-id bleibt.
  const [rev, setRev] = useState<Record<string, number>>({});
  // Zeile des aktuellen Suchtreffers — EIGENE Markierung, nicht
  // die Abspiel-Markierung: ohne sie sieht man nur, dass irgendwo
  // gescrollt wurde, und liest den Treffer in der Nachbarzeile
  // (User 2026-09-09: „liefert Workshop wenn ich Werkstatt suche" —
  // Segment 99 „Workshop-Tagen" steht direkt über Segment 100
  // „Werkstattverfahren", dem echten Treffer).
  const [suchZeile, setSuchZeile] = useState(-1);
  const audioRef = useRef<HTMLAudioElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const saveTimer = useRef<number | undefined>(undefined);
  const zustand = useRef({ sprecher, segmente });
  zustand.current = { sprecher, segmente };
  const aktivRef = useRef(aktiv);
  aktivRef.current = aktiv;

  useEffect(() => {
    void apiGet<Transkript>(`/api/transcripts/${id}`).then((t) => {
      setName(t.name);
      setSprecher(t.sprecher);
      setSegmente(t.segmente.map((s) => ({ ...s,
        id: s.id || neueId() })));
      setHatAudio(!!t.audio);
    }).catch((e) => setFehler(errMsg(e)));
  }, [id]);

  const speichere = useCallback(async () => {
    setSpeichert(true);
    try {
      await apiSend<{ updated: string }>(
        `/api/transcripts/${id}`,
        { sprecher: zustand.current.sprecher,
          segmente: zustand.current.segmente }, "PUT");
      // Lokalzeit, hh:mm:ss (Review: UTC-Slice zeigte falsche Uhrzeit)
      setGespeichert(new Date().toLocaleTimeString([], {
        hour: "2-digit", minute: "2-digit", second: "2-digit" }));
      setFehler("");
    } catch (e) {
      setFehler(tr("ed.speicherfehler", { e: errMsg(e) }));
    } finally { setSpeichert(false); }
  }, [id, tr]);

  const ausstehend = useRef(false);
  const speichereRef = useRef(speichere);
  speichereRef.current = speichere;
  const dirty = useCallback(() => {
    ausstehend.current = true;
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => {
      ausstehend.current = false;
      void speichere();
    }, 1200);
  }, [speichere]);
  // Review-Befund (HOCH): der Unmount-Cleanup verwarf den anstehenden
  // Save — Zurück/Tab-Wechsel innerhalb der Debounce verlor den
  // letzten Edit still. Jetzt: ausstehenden Save FLUSHEN (speichere
  // liest zustand.current und ist damit unmount-sicher).
  useEffect(() => () => {
    window.clearTimeout(saveTimer.current);
    if (ausstehend.current) {
      ausstehend.current = false;
      void speichereRef.current();
    }
  }, []);

  // ---------- Segment-Ops ----------
  const textAendern = useCallback((sid: string, text: string) => {
    setSegmente((s) => s.map((x) => x.id === sid ? { ...x, text } : x));
    dirty();
  }, [dirty]);

  const sprecherSetzen = useCallback((sid: string,
                                      wer: string | null) => {
    setSegmente((s) => s.map((x) => x.id === sid
      ? { ...x, sprecher: wer } : x));
    dirty();
  }, [dirty]);

  const teilen = useCallback((sid: string, cursor: number) => {
    setSegmente((s) => {
      const i = s.findIndex((x) => x.id === sid);
      if (i < 0) return s;
      const seg = s[i];
      const a = seg.text.slice(0, cursor).trim();
      const b = seg.text.slice(cursor).trim();
      if (!a || !b) return s;
      const anteil = a.length / (a.length + b.length);
      const mitte = seg.start + (seg.end - seg.start) * anteil;
      // beide Hälften mit NEUER id — unkontrollierte Textareas
      // (defaultValue) zeigen neuen Text nur nach Remount
      const neu: Segment[] = [
        { ...seg, id: neueId(),
          end: Math.round(mitte * 1000) / 1000, text: a },
        { id: neueId(), start: Math.round(mitte * 1000) / 1000,
          end: seg.end, sprecher: seg.sprecher, text: b }];
      return [...s.slice(0, i), ...neu, ...s.slice(i + 1)];
    });
    dirty();
  }, [dirty]);

  const verbinden = useCallback((sid: string) => {
    setSegmente((s) => {
      const i = s.findIndex((x) => x.id === sid);
      if (i < 1) return s;
      const prev = s[i - 1], seg = s[i];
      // NEUE id: die Textarea ist unkontrolliert (defaultValue) und
      // zeigt den zusammengeführten Text nur nach Remount
      const zusammen = { ...prev, id: neueId(), end: seg.end,
        text: `${prev.text} ${seg.text}`.trim() };
      return [...s.slice(0, i - 1), zusammen, ...s.slice(i + 1)];
    });
    dirty();
  }, [dirty]);

  const entfernen = useCallback((sid: string) => {
    setSegmente((s) => s.filter((x) => x.id !== sid));
    dirty();
  }, [dirty]);

  // ---------- Sprecher-Ops ----------
  const umbenennen = useCallback((wer: string, neuName: string) => {
    setSprecher((sp) => sp.map((x) => x.id === wer
      ? { ...x, name: neuName } : x));
    dirty();
  }, [dirty]);

  const sprecherNeu = useCallback(() => {
    setSprecher((sp) => {
      const n = sp.length + 1;
      return [...sp, { id: `sp${Date.now().toString(36)}`,
        name: `${tr("ed.sprecher")} ${n}` }];
    });
    dirty();
  }, [dirty, tr]);

  const zusammenfuehren = useCallback((von: string, nach: string) => {
    setSegmente((s) => s.map((x) => x.sprecher === von
      ? { ...x, sprecher: nach } : x));
    setSprecher((sp) => sp.filter((x) => x.id !== von));
    dirty();
  }, [dirty]);

  const leereZuweisen = useCallback((wer: string) => {
    setSegmente((s) => s.map((x) => x.sprecher
      ? x : { ...x, sprecher: wer }));
    dirty();
  }, [dirty]);

  // ---------- Suchen & Ersetzen ----------
  const zeigeTreffer = useCallback((i: number) => {
    setSuchZeile(i);
    if (i < 0) return;
    setAktiv(i);
    listRef.current?.querySelector(`[data-seg="${i}"]`)
      ?.scrollIntoView({ block: "center", behavior: "smooth" });
    // Der Kopf springt mit — aber nur im Stillstand; in laufender
    // Wiedergabe würde die Suche das Hören unterbrechen.
    const a = audioRef.current;
    const seg = zustand.current.segmente[i];
    if (a && a.paused && seg) a.currentTime = seg.start;
  }, []);

  const textErsetzen = useCallback((sid: string, off: number,
                                    laenge: number, ersatz: string) => {
    setSegmente((s) => s.map((x) => x.id === sid
      ? { ...x, text: x.text.slice(0, off) + ersatz
                      + x.text.slice(off + laenge) }
      : x));
    setRev((r) => ({ ...r, [sid]: (r[sid] ?? 0) + 1 }));
    dirty();
  }, [dirty]);

  const alleErsetzen = useCallback((was: string, ersatz: string,
                                    gross: boolean,
                                    weich: boolean) => {
    // aus zustand.current gerechnet, NICHT im State-Updater: der wird
    // im StrictMode doppelt aufgerufen und würde doppelt zählen
    const segs = zustand.current.segmente;
    const bump: Record<string, true> = {};
    let n = 0;
    const neuSegs = segs.map((x) => {
      const [text, k] = ersetzeAlleIn(x.text, was, ersatz,
                                      gross, weich);
      if (!k) return x;
      n += k;
      bump[x.id] = true;
      return { ...x, text };
    });
    if (!n) return 0;
    setSegmente(neuSegs);
    setRev((r) => {
      const o = { ...r };
      for (const k of Object.keys(bump)) o[k] = (o[k] ?? 0) + 1;
      return o;
    });
    dirty();
    return n;
  }, [dirty]);

  // ---------- Player ----------
  const starts = useMemo(() => segmente.map((s) => s.start),
                         [segmente]);
  const sprecherName = useMemo(
    () => new Map(sprecher.map((s) => [s.id, s.name])), [sprecher]);
  const indexBei = useCallback((t: number) => {
    let lo = 0, hi = starts.length - 1, aus = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (starts[mid] <= t) { aus = mid; lo = mid + 1; }
      else hi = mid - 1;
    }
    return aus;
  }, [starts]);

  const onTime = useCallback(() => {
    const a = audioRef.current;
    if (!a) return;
    if (loop && aktiv >= 0 && segmente[aktiv]
        && a.currentTime > segmente[aktiv].end - 0.04) {
      a.currentTime = segmente[aktiv].start;
      return;
    }
    const i = indexBei(a.currentTime);
    if (i !== aktiv) {
      setAktiv(i);
      if (folgen && i >= 0) {
        listRef.current?.querySelector(`[data-seg="${i}"]`)
          ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }
  }, [aktiv, folgen, indexBei, loop, segmente]);

  const springe = useCallback((t: number, abspielen = false) => {
    const a = audioRef.current;
    if (!a) return;
    a.currentTime = Math.max(0, t);
    if (abspielen) void a.play();
  }, []);

  useEffect(() => {
    const a = audioRef.current;
    if (a) a.playbackRate = speed;
    lset(KEYS.editorSpeed, String(speed));
  }, [speed, hatAudio]);

  useEffect(() => {
    // Tastatur-Schema v2 (User 2026-08-30 — Ctrl+Pfeile frisst
    // macOS/Mission Control!): J/K/L-Shuttle wie Audition; mit ⌥
    // auch MITTEN IM TIPPEN (e.code, weil ⌥+Taste auf macOS das
    // komponierte Zeichen in e.key legt); außerhalb der Textfelder
    // zusätzlich pur J/K/L, ←/→ und Leertaste wie QuickTime.
    const h = (e: KeyboardEvent) => {
      const ziel = e.target as HTMLElement | null;
      const tippt = ziel?.tagName === "TEXTAREA"
        || ziel?.tagName === "INPUT";
      const a = audioRef.current;
      if (!a) return;
      const toggle = () => {
        if (a.paused) void a.play(); else a.pause();
      };
      if (e.altKey && !e.metaKey && !e.ctrlKey) {
        if (e.code === "KeyJ") { e.preventDefault(); a.currentTime -= 5; }
        else if (e.code === "KeyL") {
          e.preventDefault(); a.currentTime += 5;
        } else if (e.code === "KeyK") { e.preventDefault(); toggle(); }
        return;
      }
      if (e.ctrlKey && !e.metaKey && !e.altKey) {
        if (e.code === "KeyL") { e.preventDefault(); setLoop((l) => !l); }
        else if (e.code === "KeyX") {
          e.preventDefault();
          setSpeed((s) => SPEEDS[(SPEEDS.indexOf(s) + 1)
            % SPEEDS.length]);
        }
        return;
      }
      if (tippt || e.metaKey) {
        // Shift+Space als Alt-Weg außerhalb der Felder (v1-Erbe)
        return;
      }
      if (e.code === "KeyJ" || e.key === "ArrowLeft") {
        e.preventDefault(); a.currentTime -= 5;
      } else if (e.code === "KeyL" || e.key === "ArrowRight") {
        e.preventDefault(); a.currentTime += 5;
      } else if (e.code === "KeyK" || e.key === " ") {
        e.preventDefault(); toggle();
      } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        // Absatz-Schritt (User 2026-08-30): ↑/↓ laufen die Segmente
        // entlang — Audio auf den Segment-Anfang, Zeile aktiv+scrollen
        e.preventDefault();
        const segs = zustand.current.segmente;
        if (!segs.length) return;
        const cur = aktivRef.current;
        const i = e.key === "ArrowDown"
          ? Math.min(cur < 0 ? 0 : cur + 1, segs.length - 1)
          : Math.max(cur < 0 ? 0 : cur - 1, 0);
        a.currentTime = segs[i].start;
        setAktiv(i);
        listRef.current?.querySelector(`[data-seg="${i}"]`)
          ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    };
    document.addEventListener("keydown", h);
    return () => document.removeEventListener("keydown", h);
  }, []);

  // EIN geteiltes Sprecher-Menü für alle Zeilen (PERF-Umbau)
  const [menue, setMenue] = useState<{ segId: string; x: number;
    y: number } | null>(null);
  const menueOeffnen = useCallback((segId: string, x: number,
                                   y: number) => {
    setMenue({ segId, x, y });
  }, []);
  useEffect(() => {
    if (!menue) return;
    const zu = (e: Event) => {
      if ((e.target as HTMLElement | null)
          ?.closest?.("[data-sprecher-menue]")) return;
      setMenue(null);
    };
    document.addEventListener("pointerdown", zu, true);
    const esc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenue(null);
    };
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", zu, true);
      document.removeEventListener("keydown", esc);
    };
  }, [menue]);

  // ---------- Export ----------
  const [exportNote, setExportNote] = useState("");
  const exportiere = useCallback(async (format: string) => {
    setExportNote("");
    const endung = format === "enrich" ? "enrich.zip"
      : format === "qdpx" ? "qdpx.zip" : format;
    try {
      if (isTauri()) {
        const p = await savePath(`${name || "transkript"}.${endung}`,
                                 format === "enrich" || format === "qdpx"
                                   ? "zip" : format);
        if (!p) return;
        await apiSend(`/api/transcripts/${id}/export`,
                      { format, path: p });
        setExportNote(tr("ed.exportiert", { p }));
      } else {
        window.open(`${API_BASE}/api/transcripts/${id}/export/${format}`,
                    "_blank");
      }
    } catch (e) {
      setExportNote(tr("ed.exportfehler", { e: errMsg(e) }));
    }
  }, [id, name, tr]);

  return (
    <Flex style={{ height: "100%", minHeight: 0 }}>
      <Flex direction="column"
            style={{ flex: 1, minWidth: 0, minHeight: 0 }}>
        <Flex align="center" gap="2" px="4" py="2"
              style={{ borderBottom: "1px solid var(--gray-a5)" }}>
          <Button size="1" variant="ghost" onClick={onExit}>
            <Icon name="back" /> {tr("ed.zurueck")}</Button>
          <Text size="2" weight="medium" truncate
                style={{ flex: 1, minWidth: 0 }}>{kuerze(name, 80)}</Text>
          <Text size="1" color="gray">
            {speichert ? tr("ed.speichert")
              : gespeichert
                ? tr("ed.gespeichert", { t: gespeichert }) : ""}
          </Text>
          <ExportMenu onExport={exportiere} />
        </Flex>
        {fehler && <ErrorNote>{fehler}</ErrorNote>}
        {exportNote && (
          <Text size="1" color="gray"
                style={{ padding: "4px 16px" }}>{exportNote}</Text>
        )}

        <div ref={listRef}
             style={{ flex: 1, overflowY: "auto", minHeight: 0,
                      padding: "8px 16px 96px" }}>
          {segmente.length === 0 && (
            <Text size="2" color="gray">{tr("ed.leer")}</Text>
          )}
          {segmente.map((seg, i) => (
            <SegmentZeile key={`${seg.id}#${rev[seg.id] ?? 0}`}
              seg={seg} index={i}
              aktiv={i === aktiv}
              treffer={i === suchZeile}
              name={sprecherName.get(seg.sprecher ?? "") ?? ""}
              farbe={sprecherFarbe(sprecher, seg.sprecher)}
              hatAudio={hatAudio}
              onSpringe={springe}
              onText={textAendern} onMenue={menueOeffnen}
              onTeilen={teilen} onVerbinden={verbinden}
              onEntfernen={entfernen} />
          ))}
        </div>

        {/* Drei Zonen (User 2026-09-09): der Transport steht MITTIG in
            der Spalte, die Laufzeit rechts — die beiden Randzonen sind
            gleich breit (flex 1), damit die Mitte echt die Mitte ist. */}
        <Flex align="center" gap="2" px="4" py="2"
              style={{ borderTop: "1px solid var(--gray-a5)",
                       background: "var(--color-panel-solid)" }}>
          {hatAudio ? (
            <>
              <div style={{ flex: 1 }} />
              <audio ref={audioRef}
                     src={`${API_BASE}/api/transcripts/${id}/audio`}
                     onTimeUpdate={onTime}
                     onPlay={(e) => { setLaeuft(true);
                       e.currentTarget.playbackRate = speed; }}
                     onPause={() => setLaeuft(false)} />
              <IconButton title={tr("ed.rueck5")} onClick={() => {
                if (audioRef.current)
                  audioRef.current.currentTime -= 5;
              }}><Icon name="rewind" size={16} /></IconButton>
              <IconButton title={laeuft ? tr("ed.pause") : tr("ed.play")}
                          onClick={() => {
                const a = audioRef.current;
                if (!a) return;
                if (a.paused) void a.play(); else a.pause();
              }}><Icon name={laeuft ? "pause" : "play"} size={18} />
              </IconButton>
              <IconButton title={tr("ed.vor5")} onClick={() => {
                if (audioRef.current)
                  audioRef.current.currentTime += 5;
              }}><Icon name="forward" size={16} /></IconButton>
              <Button size="1" variant="ghost"
                      title={tr("ed.speed")} onClick={() => {
                const i = SPEEDS.indexOf(speed);
                setSpeed(SPEEDS[(i + 1) % SPEEDS.length]);
              }}>{speed.toFixed(2).replace(/0$/, "")}×</Button>
              <IconButton title={tr("ed.loop")} onClick={() => setLoop(!loop)}>
                <span style={{ opacity: loop ? 1 : 0.4 }}>
                  <Icon name="loop" size={16} /></span>
              </IconButton>
              <label style={{ display: "flex", alignItems: "center",
                              gap: 6 }}>
                <Checkbox checked={folgen} onCheckedChange={(v) => {
                  setFolgen(v === true);
                  lset(KEYS.editorFolgen, v === true ? "1" : "0");
                }} />
                <Text size="1">{tr("ed.folgen")}</Text>
              </label>
              <Flex justify="end" align="center" style={{ flex: 1 }}>
                <Text size="1" color="gray"
                      style={{ fontVariantNumeric: "tabular-nums" }}>
                  {aktiv >= 0 && segmente[aktiv]
                    ? hms(segmente[aktiv].start) : ""}
                </Text>
              </Flex>
            </>
          ) : (
            <Text size="1" color="gray">{tr("ed.keinaudio")}</Text>
          )}
        </Flex>
      </Flex>

      {menue && (
        <div data-sprecher-menue
             style={{ position: "fixed", left: menue.x,
                      top: Math.min(menue.y, window.innerHeight - 260),
                      zIndex: 60, background: "var(--color-panel-solid)",
                      border: "1px solid var(--gray-a6)",
                      borderRadius: 8, boxShadow: "var(--shadow-4)",
                      padding: 4, minWidth: 160, maxHeight: 250,
                      overflowY: "auto" }}>
          <MenueEintrag label={tr("ed.sprecher.ohne")} farbe="gray"
                        onClick={() => { sprecherSetzen(menue.segId,
                          null); setMenue(null); }} />
          {sprecher.map((s) => (
            <MenueEintrag key={s.id} label={s.name}
                          farbe={sprecherFarbe(sprecher, s.id)}
                          onClick={() => { sprecherSetzen(menue.segId,
                            s.id); setMenue(null); }} />
          ))}
        </div>
      )}
      <SidePanel side="right" storageKey={KEYS.sidebarSprecher}
                 defaultWidth={260} resizable
                 title={
                   <SegTabs value={seitenTab}
                     onChange={(v) => {
                       sset(KEYS.editorSeitenTab, v);
                       setSeitenTab(v as "sprecher" | "suchen");
                       if (v !== "suchen") setSuchZeile(-1);
                     }}
                     options={[
                       { value: "sprecher", label: tr("ed.sprecher") },
                       { value: "suchen", label: tr("ed.tab.suchen") }]} />
                 }>
        {seitenTab === "sprecher"
          ? <SprecherPanel id={id} sprecher={sprecher}
                           segmente={segmente} hatAudio={hatAudio}
                           onRename={umbenennen} onNeu={sprecherNeu}
                           onMerge={zusammenfuehren}
                           onLeere={leereZuweisen} />
          : <SuchPanel segmente={segmente} onZeige={zeigeTreffer}
                       onErsetze={textErsetzen}
                       onAlleErsetzen={alleErsetzen} />}
      </SidePanel>
    </Flex>
  );
}

function ExportMenu({ onExport }: {
  onExport: (format: string) => void;
}) {
  const tr = useT();
  return (
    <Select.Root value="" onValueChange={(v) => v && onExport(v)}>
      <Select.Trigger placeholder={tr("ed.export")} variant="soft" />
      <Select.Content>
        <Select.Item value="vtt">VTT</Select.Item>
        <Select.Item value="csv">CSV</Select.Item>
        <Select.Item value="txt">TXT</Select.Item>
        <Select.Item value="enrich">{tr("ed.export.enrich")}
        </Select.Item>
        <Select.Item value="qdpx">{tr("ed.export.qdpx")}
        </Select.Item>
      </Select.Content>
    </Select.Root>
  );
}

function MenueEintrag({ label, farbe, onClick }: {
  label: string; farbe: string; onClick: () => void;
}) {
  return (
    <button type="button" onClick={onClick}
            style={{ display: "block", width: "100%",
                     textAlign: "left", background: "none",
                     border: "none", padding: "5px 8px",
                     borderRadius: 6, cursor: "pointer",
                     font: "inherit", fontSize: 13 }}
            onMouseEnter={(e) => e.currentTarget.style.background
              = "var(--gray-a3)"}
            onMouseLeave={(e) => e.currentTarget.style.background
              = "none"}>
      <Badge color={farbe as never} variant="soft">{label}</Badge>
    </button>
  );
}

// PERF (Live-Befund „jeder Buchstabe 2 s"): die Zeile ist memoisiert
// mit EIGENEM Vergleich — Eltern-Renders (Tippen im Sprecher-Panel,
// Autosave-Status, aktiv-Wechsel) erreichen nur Zeilen, deren
// abgeleitete Props (seg/name/farbe/aktiv) sich wirklich ändern.
// `liste` (Dropdown-Inhalt) ist BEWUSST vom Vergleich ausgenommen;
// die Textarea misst ihre Höhe nur bei Mount + Eingabe (erzwungenes
// Layout je Render war der Haupt-Kostenpunkt × 558 Zeilen).
const SegmentZeile = memo(function SegmentZeile({
  seg, index, aktiv, treffer, name, farbe, hatAudio, onSpringe, onText,
  onMenue, onTeilen, onVerbinden, onEntfernen,
}: {
  seg: Segment; index: number; aktiv: boolean;
  /** aktueller Suchtreffer — Ring statt Füllung, damit er
      von der Abspiel-Markierung unterscheidbar bleibt */
  treffer: boolean; name: string;
  farbe: ReturnType<typeof sprecherFarbe>;
  hatAudio: boolean;
  onSpringe: (t: number, abspielen?: boolean) => void;
  onText: (id: string, text: string) => void;
  onMenue: (segId: string, x: number, y: number) => void;
  onTeilen: (id: string, cursor: number) => void;
  onVerbinden: (id: string) => void;
  onEntfernen: (id: string) => void;
}) {
  const tr = useT();
  const taRef = useRef<HTMLTextAreaElement | null>(null);

  const wachsen = (el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    // +2 = die beiden Rahmen (border-box); ohne sie ist die Zeile
    // flacher als SEG_SLOT und die Grundlinien laufen auseinander
    el.style.height = `${el.scrollHeight + 2}px`;
  };

  return (
    <div data-seg={index}
         style={{
           display: "grid",
           gridTemplateColumns: "26px 74px 130px 1fr 76px",
           gap: 8, alignItems: "start", padding: "5px 4px",
           borderRadius: 8,
           background: aktiv ? "var(--accent-a3)" : undefined,
           outline: treffer ? "2px solid var(--accent-8)" : undefined,
           outlineOffset: -2,
         }}>
      <div style={{ ...SEG_SLOT, justifyContent: "center" }}>
        <IconButton title={tr("ed.abhier")}
                    onClick={() => onSpringe(seg.start, true)}>
          {/* display:flex nimmt dem Icon den Inline-Kontext — sonst
              zieht sein verticalAlign(-2px) den Glyph aus der Zeile */}
          <span style={{ display: "flex",
                         opacity: hatAudio ? 1 : 0.25 }}>
            <Icon name="play" size={14} /></span>
        </IconButton>
      </div>
      <div style={SEG_SLOT}>
        <Text size="1" color="gray" style={{
          cursor: hatAudio ? "pointer" : undefined,
          fontVariantNumeric: "tabular-nums" }}
              onClick={() => onSpringe(seg.start)}>{hms(seg.start)}</Text>
      </div>
      {/* leichter Knopf statt Radix-Select je Zeile (PERF: ~6 ms ×
          557 Zeilen je Render) — EIN geteiltes Menü im Parent */}
      <button type="button"
              onClick={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                onMenue(seg.id, r.left, r.bottom + 2);
              }}
              style={{ ...SEG_SLOT, background: "none", border: "none",
                       padding: 0, textAlign: "left", cursor: "pointer",
                       maxWidth: 130, overflow: "hidden" }}>
        <Badge color={farbe} variant="soft">
          {name || tr("ed.sprecher.ohne")}
        </Badge>
      </button>
      <textarea ref={(el) => {
                  taRef.current = el;
                  // Höhe NUR beim ersten Anhängen messen — je Render
                  // wäre es ein erzwungenes Layout pro Zeile
                  if (el && el.dataset.auto !== "1") {
                    el.dataset.auto = "1";
                    wachsen(el);
                  }
                }}
                defaultValue={seg.text}
                rows={1}
                onInput={(e) => {
                  wachsen(e.currentTarget);
                  onText(seg.id, e.currentTarget.value);
                }}
                className="seg-text" />
      <Flex gap="1" style={SEG_SLOT}>
        <IconButton title={tr("ed.teilen")} onClick={() => {
          const pos = taRef.current?.selectionStart ?? 0;
          onTeilen(seg.id, pos);
        }}><Icon name="split" size={14} /></IconButton>
        <IconButton title={tr("ed.verbinden")}
                    onClick={() => onVerbinden(seg.id)}>
          <Icon name="merge" size={14} /></IconButton>
        <IconButton title={tr("ed.zeile.loeschen")}
                    onClick={() => onEntfernen(seg.id)}>
          <Icon name="trash" size={14} /></IconButton>
      </Flex>
    </div>
  );
}, (a, b) => a.seg === b.seg && a.aktiv === b.aktiv
  && a.treffer === b.treffer
  && a.index === b.index && a.name === b.name && a.farbe === b.farbe
  && a.hatAudio === b.hatAudio);

// Tippen bleibt LOKAL (nur dieses Feld rendert), der Commit in den
// globalen State läuft debounced — sonst rendert jeder Buchstabe alle
// Zeilen des Sprechers neu (Live-Befund: 2 s je Taste bei 557 Zeilen).
function NameFeld({ id, name, onRename }: {
  id: string; name: string;
  onRename: (id: string, name: string) => void;
}) {
  const [wert, setWert] = useState(name);
  const timer = useRef<number | undefined>(undefined);
  const letzte = useRef(name);
  useEffect(() => {
    if (name !== letzte.current) { setWert(name); letzte.current = name; }
  }, [name]);
  const commit = (v: string) => {
    letzte.current = v;
    onRename(id, v);
  };
  return (
    <TextField.Root size="1" value={wert} style={{ flex: 1 }}
      onChange={(e) => {
        setWert(e.target.value);
        window.clearTimeout(timer.current);
        const v = e.target.value;
        timer.current = window.setTimeout(() => commit(v), 500);
      }}
      onBlur={() => {
        window.clearTimeout(timer.current);
        if (wert !== letzte.current) commit(wert);
      }} />
  );
}

function SprecherPanel({ id, sprecher, segmente, hatAudio, onRename,
                         onNeu, onMerge, onLeere }: {
  id: string; sprecher: Sprecher[]; segmente: Segment[];
  hatAudio: boolean;
  onRename: (id: string, name: string) => void;
  onNeu: () => void;
  onMerge: (von: string, nach: string) => void;
  onLeere: (wer: string) => void;
}) {
  const tr = useT();
  const ohne = segmente.filter((s) => !s.sprecher).length;
  const sample = (sid: string) => {
    const a = new Audio(
      `${API_BASE}/api/transcripts/${id}/sprecher/${sid}/sample`);
    void a.play();
  };
  // Zusammenführen als Icon-Knopf mit Klappmenü (Layout-Befund
  // 2026-09-09): der breite Select-Platzhalter „Zusammenführen in …"
  // sprengte die schmale Sidebar — die Segment-Zahl brach um. Jetzt
  // dasselbe Menü-Muster wie in den Segment-Zeilen, die Aktionszeile
  // bleibt bei jeder Panel-Breite einzeilig.
  const [merge, setMerge] = useState<string | null>(null);
  useEffect(() => {
    if (!merge) return;
    const zu = (e: Event) => {
      if ((e.target as HTMLElement | null)
          ?.closest?.("[data-merge-menue]")) return;
      setMerge(null);
    };
    document.addEventListener("pointerdown", zu, true);
    const esc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMerge(null);
    };
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", zu, true);
      document.removeEventListener("keydown", esc);
    };
  }, [merge]);
  return (
    // EIN Rasterrand für die ganze App (User 2026-09-09): 16 px —
    // dieselbe Kante wie „LocalTranscript" links und der
    // Einstellungen-Knopf rechts. Ghost-Knöpfe tragen negative
    // Ränder, ihr GLYPH sitzt damit ebenfalls auf 16 px.
    <Flex direction="column" gap="2" px="4" py="3">
      {sprecher.map((s) => {
        const n = segmente.filter((x) => x.sprecher === s.id).length;
        return (
          <Flex key={s.id} direction="column" gap="1"
                style={{ position: "relative",
                         borderBottom: "1px solid var(--gray-a4)",
                         paddingBottom: 8 }}>
            <Flex align="center" gap="2">
              <Badge color={sprecherFarbe(sprecher, s.id)}
                     variant="solid" radius="full"> </Badge>
              <NameFeld id={s.id} name={s.name} onRename={onRename} />
            </Flex>
            <Flex align="center" gap="1">
              <Text size="1" color="gray"
                    style={{ whiteSpace: "nowrap" }}>
                {tr("ed.sprecher.n", { n })}</Text>
              <div style={{ flex: 1 }} />
              {hatAudio && n > 0 && (
                <IconButton title={tr("ed.sprecher.probe")}
                            onClick={() => sample(s.id)}>
                  <Icon name="sample" size={14} /></IconButton>
              )}
              {sprecher.length > 1 && (
                <IconButton title={tr("ed.sprecher.merge")}
                            onClick={() => setMerge(
                              (cur) => cur === s.id ? null : s.id)}>
                  <Icon name="merge" size={14} /></IconButton>
              )}
            </Flex>
            {merge === s.id && (
              <div data-merge-menue
                   style={{ position: "absolute", right: 0, top: "100%",
                            zIndex: 60, minWidth: 160, maxWidth: "100%",
                            maxHeight: 250, overflowY: "auto",
                            padding: 4, borderRadius: 8,
                            background: "var(--color-panel-solid)",
                            border: "1px solid var(--gray-a6)",
                            boxShadow: "var(--shadow-4)" }}>
                <Text size="1" color="gray" as="div"
                      style={{ padding: "2px 8px 4px" }}>
                  {tr("ed.sprecher.merge")}</Text>
                {sprecher.filter((x) => x.id !== s.id).map((x) => (
                  <MenueEintrag key={x.id} label={x.name}
                                farbe={sprecherFarbe(sprecher, x.id)}
                                onClick={() => { setMerge(null);
                                  onMerge(s.id, x.id); }} />
                ))}
              </div>
            )}
            {ohne > 0 && (
              <Button size="1" variant="ghost"
                      onClick={() => onLeere(s.id)}>
                {tr("ed.sprecher.leere")} ({ohne})</Button>
            )}
          </Flex>
        );
      })}
      <Button size="1" variant="soft" onClick={onNeu}>
        <Icon name="plus" size={14} /> {tr("ed.sprecher.neu")}</Button>
    </Flex>
  );
}

// ---------- Suchen & Ersetzen (Seitenspalte, Tab 2) ----------

/** Passt `nadel` ab Position i? Beide Zeichenketten kommen bereits
    normalisiert (Groß-/Kleinschreibung) herein. `weich` überliest
    einen TRENNSTRICH samt folgender Leerzeichen/Umbrüche mitten im
    Wort — so findet „Werkstatt" auch „Werk- statt". Gibt die Länge
    IM TEXT zurück (kann länger sein als die Nadel) oder -1. */
function passtAb(heu: string, nadel: string, i: number,
                 weich: boolean): number {
  let j = i, k = 0;
  while (k < nadel.length) {
    if (j >= heu.length) return -1;
    if (weich && k > 0 && heu[j] === "-") {
      let m = j + 1;
      while (m < heu.length && /\s/.test(heu[m])) m += 1;
      if (m < heu.length && heu[m] === nadel[k]) { j = m; continue; }
      return -1;
    }
    if (heu[j] !== nadel[k]) return -1;
    j += 1; k += 1;
  }
  return j - i;
}

/** Alle Vorkommen in EINEM Text — LITERAL: kein Wörterbuch, keine
    Stammformen, keine Übersetzung, und ohne RegExp, damit Sonder-
    zeichen im Suchbegriff (. * ? [ ) nichts kaputtmachen. */
function findeAlle(text: string, was: string, gross: boolean,
                   weich: boolean): { off: number; len: number }[] {
  const aus: { off: number; len: number }[] = [];
  if (!was) return aus;
  const heu = gross ? text : text.toLowerCase();
  const nadel = gross ? was : was.toLowerCase();
  if (!weich) {
    let p = heu.indexOf(nadel);
    while (p >= 0) {
      aus.push({ off: p, len: nadel.length });
      p = heu.indexOf(nadel, p + nadel.length);
    }
    return aus;
  }
  // weich: nur dort genau prüfen, wo das erste Zeichen sitzt —
  // sonst wäre es O(Text × Nadel) bei jedem Tastendruck
  let i = heu.indexOf(nadel[0]);
  while (i >= 0) {
    const len = passtAb(heu, nadel, i, true);
    if (len > 0) {
      aus.push({ off: i, len });
      i = heu.indexOf(nadel[0], i + len);
    } else {
      i = heu.indexOf(nadel[0], i + 1);
    }
  }
  return aus;
}

/** Alle Vorkommen in EINEM Text ersetzen; gibt neuen Text + Anzahl. */
function ersetzeAlleIn(text: string, was: string, ersatz: string,
                       gross: boolean,
                       weich: boolean): [string, number] {
  const tr = findeAlle(text, was, gross, weich);
  if (!tr.length) return [text, 0];
  let aus = "", p = 0;
  for (const x of tr) {
    aus += text.slice(p, x.off) + ersatz;
    p = x.off + x.len;
  }
  return [aus + text.slice(p), tr.length];
}

// `len` ist die Länge IM TEXT — bei überlesener Trennung länger als
// der Suchbegriff („Werk- statt" = 11 für „Werkstatt" = 9)
type Treffer = { i: number; seg: string; off: number; len: number };

function SuchPanel({ segmente, onZeige, onErsetze, onAlleErsetzen }: {
  segmente: Segment[];
  onZeige: (i: number) => void;
  onErsetze: (sid: string, off: number, laenge: number,
              ersatz: string) => void;
  onAlleErsetzen: (was: string, ersatz: string,
                   gross: boolean, weich: boolean) => number;
}) {
  const tr = useT();
  const [was, setWas] = useState("");
  const [womit, setWomit] = useState("");
  const [gross, setGross] = useState(false);
  const [weich, setWeich] = useState(true);
  const [idx, setIdx] = useState(0);
  const [note, setNote] = useState("");

  const treffer = useMemo<Treffer[]>(() => {
    if (!was) return [];
    const aus: Treffer[] = [];
    segmente.forEach((s, i) => {
      for (const x of findeAlle(s.text, was, gross, weich)) {
        aus.push({ i, seg: s.id, off: x.off, len: x.len });
      }
    });
    return aus;
  }, [segmente, was, gross, weich]);

  const zeigeRef = useRef(onZeige);
  zeigeRef.current = onZeige;
  const trefferRef = useRef(treffer);
  trefferRef.current = treffer;

  // Neue Suche: zählen und zum ERSTEN Vorkommen springen. Hängt
  // bewusst nur an der Anfrage — nicht an `treffer`, sonst würde
  // jeder Tastendruck im Transkript zurück an den Anfang springen.
  useEffect(() => {
    setIdx(0);
    setNote("");
    const t = trefferRef.current;
    zeigeRef.current(t.length ? t[0].i : -1);
  }, [was, gross, weich]);

  // Nach einem Ersetzen: zum nächsten Vorkommen AB der Schnittmarke —
  // so wird ein Ersatz, der den Suchbegriff enthält, nicht endlos
  // wieder gefunden.
  const weiterAb = useRef<{ i: number; off: number } | null>(null);
  useEffect(() => {
    const m = weiterAb.current;
    if (!m) return;
    weiterAb.current = null;
    const k = treffer.findIndex((x) => x.i > m.i
      || (x.i === m.i && x.off >= m.off));
    const ziel = k >= 0 ? k : 0;
    setIdx(ziel);
    if (treffer[ziel]) zeigeRef.current(treffer[ziel].i);
  }, [treffer]);

  const stelle = treffer.length
    ? Math.min(idx, treffer.length - 1) : -1;
  const cur = stelle >= 0 ? treffer[stelle] : null;

  const springe = (k: number) => {
    if (!treffer.length) return;
    const n = ((k % treffer.length) + treffer.length) % treffer.length;
    setIdx(n);
    onZeige(treffer[n].i);
  };

  const ersetzen = () => {
    if (!cur) return;
    setNote("");
    weiterAb.current = { i: cur.i, off: cur.off + womit.length };
    onErsetze(cur.seg, cur.off, cur.len, womit);
  };

  const alle = () => {
    const n = onAlleErsetzen(was, womit, gross, weich);
    setIdx(0);
    setNote(n ? tr("ed.suche.ersetzt", { n }) : "");
  };

  // Umfeld des aktuellen Treffers — zeigt IM PANEL, was gleich
  // ersetzt wird (der Fokus bleibt im Suchfeld, die Textarea im
  // Transkript wird nicht angefasst).
  const vorschau = () => {
    if (!cur) return null;
    const text = segmente[cur.i]?.text ?? "";
    const a = Math.max(0, cur.off - 26);
    const b = cur.off + cur.len;
    return (
      <Text size="1" color="gray" as="div"
            style={{ lineHeight: 1.5, wordBreak: "break-word" }}>
        {a > 0 ? "… " : ""}{text.slice(a, cur.off)}
        <mark style={{ background: "var(--accent-a4)",
                       color: "var(--gray-12)", borderRadius: 3,
                       padding: "0 1px" }}>
          {text.slice(cur.off, b)}</mark>
        {text.slice(b, b + 26)}{b + 26 < text.length ? " …" : ""}
      </Text>
    );
  };

  return (
    <Flex direction="column" gap="3" px="4" py="3">
      <Flex direction="column" gap="1">
        <Text size="1" color="gray">{tr("ed.suche.was")}</Text>
        <SearchField value={was} onChange={setWas} placeholder="" />
      </Flex>
      <Flex direction="column" gap="1">
        <Text size="1" color="gray">{tr("ed.suche.womit")}</Text>
        <TextField.Root value={womit}
                        onChange={(e) => setWomit(e.target.value)} />
      </Flex>
      <Flex direction="column" gap="2">
        <label style={{ display: "flex", alignItems: "center",
                        gap: 8 }}>
          <Checkbox checked={gross}
                    onCheckedChange={(v) => setGross(v === true)} />
          <Text size="1">{tr("ed.suche.gross")}</Text>
        </label>
        <label style={{ display: "flex", alignItems: "center",
                        gap: 8 }}>
          <Checkbox checked={weich}
                    onCheckedChange={(v) => setWeich(v === true)} />
          <Text size="1">{tr("ed.suche.weich")}</Text>
        </label>
        <Text size="1" color="gray">{tr("ed.suche.literal")}</Text>
      </Flex>

      <Flex align="center" gap="2" style={{ minHeight: 24 }}>
        <Text size="1" weight="medium"
              style={{ fontVariantNumeric: "tabular-nums" }}>
          {!was ? "" : treffer.length
            ? tr("ed.suche.stand", { i: stelle + 1, n: treffer.length })
            : tr("ed.suche.keine")}
        </Text>
        <div style={{ flex: 1 }} />
        {treffer.length > 1 && (
          <>
            <IconButton title={tr("ed.suche.zurueck")}
                        onClick={() => springe(stelle - 1)}>
              <Icon name="back" size={16} /></IconButton>
            <IconButton title={tr("ed.suche.weiter")}
                        onClick={() => springe(stelle + 1)}>
              <Icon name="next" size={16} /></IconButton>
          </>
        )}
      </Flex>
      {cur && vorschau()}
      {note && <Text size="1" color="gray">{note}</Text>}

      {/* „Alle ersetzen" steht bewusst allein in der zweiten Reihe —
          es ist die einzige Aktion, die man nicht Treffer für Treffer
          zurücknehmen kann (rückholbar nur über history/). */}
      <Flex direction="column" gap="2" align="start">
        <Flex gap="2" align="center">
          <Button size="1" variant="soft" disabled={!cur}
                  onClick={ersetzen}>{tr("ed.suche.ersetzen")}</Button>
          <Button size="1" variant="ghost" disabled={treffer.length < 2}
                  onClick={() => springe(stelle + 1)}>
            {tr("ed.suche.skip")}</Button>
        </Flex>
        <Button size="1" variant="ghost" disabled={!treffer.length}
                onClick={alle}>{tr("ed.suche.alle")}</Button>
      </Flex>
    </Flex>
  );
}
