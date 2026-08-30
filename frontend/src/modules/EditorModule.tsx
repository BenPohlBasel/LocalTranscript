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
  API_BASE, apiGet, apiSend, errMsg, hms, sprecherFarbe,
  type Segment, type Sprecher, type Transkript,
} from "../lib/api";
import { useT } from "../lib/i18n";
import { KEYS, lget, lset } from "../lib/storage";
import { isTauri, savePath } from "../lib/tauri";

const SPEEDS = [1, 1.25, 1.5, 1.75, 2];
const OHNE = " ohne";  // Select-Sentinel für "kein Sprecher"

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
      const r = await apiSend<{ updated: string }>(
        `/api/transcripts/${id}`,
        { sprecher: zustand.current.sprecher,
          segmente: zustand.current.segmente }, "PUT");
      setGespeichert(r.updated.slice(11, 16));
      setFehler("");
    } catch (e) {
      setFehler(tr("ed.speicherfehler", { e: errMsg(e) }));
    } finally { setSpeichert(false); }
  }, [id, tr]);

  const dirty = useCallback(() => {
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => {
      void speichere();
    }, 1200);
  }, [speichere]);
  useEffect(() => () => window.clearTimeout(saveTimer.current), []);

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
  }, [speed]);

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
                style={{ flex: 1 }}>{name}</Text>
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
              aktiv={i === aktiv} sprecher={sprecher}
              hatAudio={hatAudio}
              onPlay={() => springe(seg.start, true)}
              onSeek={() => springe(seg.start)}
              onText={textAendern} onSprecher={sprecherSetzen}
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
                     onPlay={() => setLaeuft(true)}
                     onPause={() => setLaeuft(false)} />
              <IconButton title="-5s" onClick={() => {
                if (audioRef.current)
                  audioRef.current.currentTime -= 5;
              }}><Icon name="rewind" size={15} /></IconButton>
              <IconButton title={laeuft ? "Pause" : "Play"}
                          onClick={() => {
                const a = audioRef.current;
                if (!a) return;
                if (a.paused) void a.play(); else a.pause();
              }}><Icon name={laeuft ? "pause" : "play"} size={17} />
              </IconButton>
              <IconButton title="+5s" onClick={() => {
                if (audioRef.current)
                  audioRef.current.currentTime += 5;
              }}><Icon name="forward" size={15} /></IconButton>
              <Button size="1" variant="ghost" onClick={() => {
                const i = SPEEDS.indexOf(speed);
                setSpeed(SPEEDS[(i + 1) % SPEEDS.length]);
              }}>{speed.toFixed(2).replace(/0$/, "")}×</Button>
              <IconButton title="Loop" onClick={() => setLoop(!loop)}>
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

const SegmentZeile = memo(function SegmentZeile({
  seg, index, aktiv, sprecher, hatAudio, onPlay, onSeek, onText,
  onSprecher, onTeilen, onVerbinden, onEntfernen,
}: {
  seg: Segment; index: number; aktiv: boolean; sprecher: Sprecher[];
  hatAudio: boolean;
  onPlay: () => void; onSeek: () => void;
  onText: (id: string, text: string) => void;
  onSprecher: (id: string, wer: string | null) => void;
  onTeilen: (id: string, cursor: number) => void;
  onVerbinden: (id: string) => void;
  onEntfernen: (id: string) => void;
}) {
  const tr = useT();
  const taRef = useRef<HTMLTextAreaElement>(null);
  const farbe = sprecherFarbe(sprecher, seg.sprecher);

  const wachsen = (el: HTMLTextAreaElement | null) => {
    if (!el) return;
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
      <IconButton title={tr("ed.abhier")} onClick={onPlay}>
        <span style={{ opacity: hatAudio ? 1 : 0.25 }}>
          <Icon name="play" size={14} /></span>
      </IconButton>
      <Text size="1" color="gray" style={{ paddingTop: 5,
        cursor: hatAudio ? "pointer" : undefined,
        fontVariantNumeric: "tabular-nums" }}
            onClick={onSeek}>{hms(seg.start)}</Text>
      <Select.Root value={seg.sprecher ?? OHNE}
                   onValueChange={(v) => onSprecher(seg.id,
                     v === OHNE ? null : v)}>
        <Select.Trigger variant="ghost" style={{ maxWidth: 130 }}>
          <Badge color={farbe} variant="soft">
            {sprecher.find((s) => s.id === seg.sprecher)?.name
              ?? tr("ed.sprecher.ohne")}
          </Badge>
        </Select.Trigger>
        <Select.Content>
          <Select.Item value={OHNE}>{tr("ed.sprecher.ohne")}
          </Select.Item>
          {sprecher.map((s) => (
            <Select.Item key={s.id} value={s.id}>{s.name}
            </Select.Item>))}
        </Select.Content>
      </Select.Root>
      <textarea ref={(el) => { (taRef as
                  React.MutableRefObject<HTMLTextAreaElement | null>)
                  .current = el; wachsen(el); }}
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
});

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
              <TextField.Root size="1" value={s.name}
                              style={{ flex: 1 }}
                              onChange={(e) =>
                                onRename(s.id, e.target.value)} />
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
