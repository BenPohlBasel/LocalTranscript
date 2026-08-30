// Editor (Ebene 2, Drilldown): Modell im STATE, nie im DOM (v1-Lehre).
// Autosave (debounced) + History im Backend; Sprecher sind Entitäten
// (umbenennen überall, Segment umhängen, zusammenführen); Segment-Ops
// teilen/verbinden/löschen; Audio-Player mit Folgen-Modus.
import { memo, useCallback, useEffect, useMemo, useRef, useState }
  from "react";
import {
  Badge, Button, Checkbox, ErrorNote, Flex, IconButton,
  Select, SidePanel, Text, TextField,
} from "../components/ui";
import { Icon } from "../components/icons";
import {
  API_BASE, apiGet, apiSend, errMsg, hms, kuerze, sprecherFarbe,
  type Segment, type Sprecher, type Transkript,
} from "../lib/api";
import { useT } from "../lib/i18n";
import { KEYS, lget, lset } from "../lib/storage";
import { isTauri, savePath } from "../lib/tauri";

const SPEEDS = [1, 1.25, 1.5, 1.75, 2];

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
  const audioRef = useRef<HTMLAudioElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const saveTimer = useRef<number | undefined>(undefined);
  const zustand = useRef({ sprecher, segmente });
  zustand.current = { sprecher, segmente };

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
    const h = (e: KeyboardEvent) => {
      const ziel = e.target as HTMLElement | null;
      const tippt = ziel?.tagName === "TEXTAREA"
        || ziel?.tagName === "INPUT";
      const a = audioRef.current;
      if (!a) return;
      if (e.key === " " && e.shiftKey && !tippt) {
        e.preventDefault();
        if (a.paused) void a.play(); else a.pause();
      } else if (e.ctrlKey && e.key === "ArrowLeft") {
        e.preventDefault(); a.currentTime -= 5;
      } else if (e.ctrlKey && e.key === "ArrowRight") {
        e.preventDefault(); a.currentTime += 5;
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
    const endung = format === "enrich" ? "enrich.zip" : format;
    try {
      if (isTauri()) {
        const p = await savePath(`${name || "transkript"}.${endung}`,
                                 format === "enrich" ? "zip" : format);
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
        <Flex align="center" gap="2" px="3" py="2"
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
                style={{ padding: "4px 12px" }}>{exportNote}</Text>
        )}

        <div ref={listRef}
             style={{ flex: 1, overflowY: "auto", minHeight: 0,
                      padding: "8px 12px 96px" }}>
          {segmente.length === 0 && (
            <Text size="2" color="gray">{tr("ed.leer")}</Text>
          )}
          {segmente.map((seg, i) => (
            <SegmentZeile key={seg.id} seg={seg} index={i}
              aktiv={i === aktiv}
              name={sprecherName.get(seg.sprecher ?? "") ?? ""}
              farbe={sprecherFarbe(sprecher, seg.sprecher)}
              hatAudio={hatAudio}
              onSpringe={springe}
              onText={textAendern} onMenue={menueOeffnen}
              onTeilen={teilen} onVerbinden={verbinden}
              onEntfernen={entfernen} />
          ))}
        </div>

        <Flex align="center" gap="2" px="3" py="2"
              style={{ borderTop: "1px solid var(--gray-a5)",
                       background: "var(--color-panel-solid)" }}>
          {hatAudio ? (
            <>
              <audio ref={audioRef}
                     src={`${API_BASE}/api/transcripts/${id}/audio`}
                     onTimeUpdate={onTime}
                     onPlay={(e) => { setLaeuft(true);
                       e.currentTarget.playbackRate = speed; }}
                     onPause={() => setLaeuft(false)} />
              <IconButton title={tr("ed.rueck5")} onClick={() => {
                if (audioRef.current)
                  audioRef.current.currentTime -= 5;
              }}><Icon name="rewind" size={15} /></IconButton>
              <IconButton title={laeuft ? tr("ed.pause") : tr("ed.play")}
                          onClick={() => {
                const a = audioRef.current;
                if (!a) return;
                if (a.paused) void a.play(); else a.pause();
              }}><Icon name={laeuft ? "pause" : "play"} size={17} />
              </IconButton>
              <IconButton title={tr("ed.vor5")} onClick={() => {
                if (audioRef.current)
                  audioRef.current.currentTime += 5;
              }}><Icon name="forward" size={15} /></IconButton>
              <Button size="1" variant="ghost" onClick={() => {
                const i = SPEEDS.indexOf(speed);
                setSpeed(SPEEDS[(i + 1) % SPEEDS.length]);
              }}>{speed.toFixed(2).replace(/0$/, "")}×</Button>
              <IconButton title={tr("ed.loop")} onClick={() => setLoop(!loop)}>
                <span style={{ opacity: loop ? 1 : 0.4 }}>
                  <Icon name="loop" size={15} /></span>
              </IconButton>
              <label style={{ display: "flex", alignItems: "center",
                              gap: 6 }}>
                <Checkbox checked={folgen} onCheckedChange={(v) => {
                  setFolgen(v === true);
                  lset(KEYS.editorFolgen, v === true ? "1" : "0");
                }} />
                <Text size="1">{tr("ed.folgen")}</Text>
              </label>
              <div style={{ flex: 1 }} />
              <Text size="1" color="gray">
                {aktiv >= 0 && segmente[aktiv]
                  ? hms(segmente[aktiv].start) : ""}
              </Text>
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
      <SidePanel side="right" title={tr("ed.sprecher")}
                 storageKey={KEYS.sidebarSprecher}
                 defaultWidth={260} resizable>
        <SprecherPanel id={id} sprecher={sprecher} segmente={segmente}
                       hatAudio={hatAudio}
                       onRename={umbenennen} onNeu={sprecherNeu}
                       onMerge={zusammenfuehren}
                       onLeere={leereZuweisen} />
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
  seg, index, aktiv, name, farbe, hatAudio, onSpringe, onText,
  onMenue, onTeilen, onVerbinden, onEntfernen,
}: {
  seg: Segment; index: number; aktiv: boolean; name: string;
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
    el.style.height = `${el.scrollHeight}px`;
  };

  return (
    <div data-seg={index}
         style={{
           display: "grid",
           gridTemplateColumns: "26px 74px 130px 1fr 76px",
           gap: 8, alignItems: "start", padding: "5px 4px",
           borderRadius: 8,
           background: aktiv ? "var(--accent-a3)" : undefined,
         }}>
      <IconButton title={tr("ed.abhier")}
                  onClick={() => onSpringe(seg.start, true)}>
        <span style={{ opacity: hatAudio ? 1 : 0.25 }}>
          <Icon name="play" size={14} /></span>
      </IconButton>
      <Text size="1" color="gray" style={{ paddingTop: 5,
        cursor: hatAudio ? "pointer" : undefined,
        fontVariantNumeric: "tabular-nums" }}
            onClick={() => onSpringe(seg.start)}>{hms(seg.start)}</Text>
      {/* leichter Knopf statt Radix-Select je Zeile (PERF: ~6 ms ×
          557 Zeilen je Render) — EIN geteiltes Menü im Parent */}
      <button type="button"
              onClick={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                onMenue(seg.id, r.left, r.bottom + 2);
              }}
              style={{ background: "none", border: "none", padding: 0,
                       textAlign: "left", cursor: "pointer",
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
                style={{
                  resize: "none", border: "none",
                  background: "transparent", width: "100%",
                  font: "inherit", fontSize: 13, lineHeight: 1.5,
                  outline: "none", padding: "3px 2px",
                  color: "var(--gray-12)",
                }} />
      <Flex gap="1" style={{ paddingTop: 3 }}>
        <IconButton title={tr("ed.teilen")} onClick={() => {
          const pos = taRef.current?.selectionStart ?? 0;
          onTeilen(seg.id, pos);
        }}><Icon name="split" size={13} /></IconButton>
        <IconButton title={tr("ed.verbinden")}
                    onClick={() => onVerbinden(seg.id)}>
          <Icon name="back" size={13} /></IconButton>
        <IconButton title={tr("ed.zeile.loeschen")}
                    onClick={() => onEntfernen(seg.id)}>
          <Icon name="trash" size={13} /></IconButton>
      </Flex>
    </div>
  );
}, (a, b) => a.seg === b.seg && a.aktiv === b.aktiv
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
  return (
    <Flex direction="column" gap="2" p="2">
      {sprecher.map((s) => {
        const n = segmente.filter((x) => x.sprecher === s.id).length;
        return (
          <Flex key={s.id} direction="column" gap="1"
                style={{ borderBottom: "1px solid var(--gray-a4)",
                         paddingBottom: 8 }}>
            <Flex align="center" gap="2">
              <Badge color={sprecherFarbe(sprecher, s.id)}
                     variant="solid" radius="full"> </Badge>
              <NameFeld id={s.id} name={s.name} onRename={onRename} />
            </Flex>
            <Flex align="center" gap="2">
              <Text size="1" color="gray">
                {tr("ed.sprecher.n", { n })}</Text>
              <div style={{ flex: 1 }} />
              {hatAudio && n > 0 && (
                <IconButton title={tr("ed.sprecher.probe")}
                            onClick={() => sample(s.id)}>
                  <Icon name="sample" size={14} /></IconButton>
              )}
              {sprecher.length > 1 && (
                <Select.Root value=""
                             onValueChange={(v) => v
                               && onMerge(s.id, v)}>
                  <Select.Trigger variant="ghost"
                    placeholder={tr("ed.sprecher.merge")} />
                  <Select.Content>
                    {sprecher.filter((x) => x.id !== s.id)
                      .map((x) => (
                        <Select.Item key={x.id} value={x.id}>
                          {x.name}</Select.Item>))}
                  </Select.Content>
                </Select.Root>
              )}
            </Flex>
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
