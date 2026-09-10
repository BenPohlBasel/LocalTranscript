// Human-Editor (User 2026-08-30): Spiegel des Speicherorts — Liste
// aller Transkripte + „Transkript importieren" (holt vtt/csv UND mp3
// in den Speicherort; Klick öffnet den Editor).
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Button, EmptyState, ErrorNote, Flex, IconButton, ListRow,
  ModalDialog, Text, TextField,
} from "../components/ui";
import { Icon } from "../components/icons";
import {
  apiGet, apiSend, apiUpload, errMsg, hms, kuerze,
  type EintragMeta,
} from "../lib/api";
import { useT } from "../lib/i18n";
import { isTauri, onFileDrop, pickAudio, pickTranskript } from "../lib/tauri";

export default function HumanEditorModule({ onOpen }: {
  onOpen: (id: string) => void;
}) {
  const tr = useT();
  const [eintraege, setEintraege] = useState<EintragMeta[]>([]);
  const [fehler, setFehler] = useState("");
  const importRef = useRef<HTMLInputElement>(null);

  const lade = useCallback(async () => {
    try {
      const r = await apiGet<{ transcripts: EintragMeta[] }>(
        "/api/transcripts");
      setEintraege(r.transcripts);
    } catch (e) { setFehler(errMsg(e)); }
  }, []);
  useEffect(() => { void lade(); }, [lade]);
  // Drop auf die Liste: .enrich (Datei oder Package-Ordner), .vtt,
  // .csv — die Shell liefert Pfade, das Backend erkennt die Form.
  useEffect(() => {
    let ab: (() => void) | undefined;
    let weg = false;
    void onFileDrop((pfade) => {
      setFehler("");
      void (async () => {
        for (const p of pfade) {
          if (!/\.(enrich|vtt|webvtt|csv)$/i.test(p)) continue;
          try { await apiSend("/api/import-path", { path: p }); }
          catch (e) { setFehler(errMsg(e)); }
        }
        void lade();
      })();
    }).then((f) => { if (weg) f(); else ab = f; });
    return () => { weg = true; ab?.(); };
  }, [lade]);

  const importiere = useCallback(async () => {
    setFehler("");
    try {
      if (isTauri()) {
        const p = await pickTranskript();
        if (!p) return;
        // Ein .enrich (Datei oder Package) trägt sein Audio schon mit
        // sich — nur bei vtt/csv lohnt die Nachfrage (User 2026-09-09).
        const audio = /\.enrich$/i.test(p)
          ? null : await pickAudio(tr("he.audiowahl"), false);
        await apiSend("/api/import-path", { path: p,
          audio_path: audio?.[0] ?? null });
        void lade();
      } else {
        importRef.current?.click();
      }
    } catch (e) { setFehler(errMsg(e)); }
  }, [lade, tr]);

  return (
    <Flex direction="column" gap="3" p="4"
          style={{ height: "100%", overflowY: "auto" }}>
      <Flex gap="3" align="center" wrap="wrap">
        <Button variant="soft" onClick={() => void importiere()}>
          <Icon name="text" /> {tr("bib.import")}</Button>
        <input ref={importRef} type="file" hidden accept=".vtt,.csv,.enrich"
               onChange={(e) => {
                 const f = e.target.files?.[0];
                 e.target.value = "";
                 if (!f) return;
                 const fd = new FormData();
                 fd.append("datei", f);
                 void apiUpload("/api/import", fd)
                   .then(() => lade())
                   .catch((err) => setFehler(errMsg(err)));
               }} />
      </Flex>

      {fehler && <ErrorNote>{fehler}</ErrorNote>}

      {eintraege.length === 0
        ? <EmptyState>{tr("he.leer")}</EmptyState>
        : eintraege.map((e) => (
            <EintragZeile key={e.id} e={e} onOpen={onOpen}
                          onChanged={() => void lade()} />
          ))}
    </Flex>
  );
}

function EintragZeile({ e, onOpen, onChanged }: {
  e: EintragMeta; onOpen: (id: string) => void;
  onChanged: () => void;
}) {
  const tr = useT();
  const [frage, setFrage] = useState<"umbenennen" | "loeschen" | null>(
    null);
  const [name, setName] = useState(e.name);
  const [dialogFehler, setDialogFehler] = useState("");
  return (
    <>
      <ListRow
        leading={<Icon name="text" size={18} />}
        title={kuerze(e.name, 72)}
        meta={`${hms(e.dauer)} · ${tr("bib.sprecher.n",
          { n: e.sprecher })} · ${tr("bib.segmente.n",
          { n: e.segmente })}`}
        onClick={() => onOpen(e.id)}
        trailing={
          <Flex gap="1" onClick={(ev) => ev.stopPropagation()}>
            <IconButton title={tr("bib.umbenennen")}
                        onClick={() => setFrage("umbenennen")}>
              <Icon name="edit" size={14} /></IconButton>
            <IconButton title={tr("bib.loeschen")}
                        onClick={() => setFrage("loeschen")}>
              <Icon name="trash" size={14} /></IconButton>
          </Flex>
        } />
      <ModalDialog open={frage === "umbenennen"}
                   onOpenChange={(o) => !o && setFrage(null)}
                   title={tr("bib.umbenennen")}
                   footer={
                     <Button onClick={() => {
                       void apiSend(`/api/transcripts/${e.id}/rename`,
                                    { name })
                         .then(() => { setFrage(null); onChanged(); })
                         .catch((err) => setDialogFehler(errMsg(err)));
                     }}>{tr("allg.ok")}</Button>}>
        <TextField.Root value={name}
                        onChange={(ev) => setName(ev.target.value)} />
        {dialogFehler && (
          <Text size="1" color="red">{dialogFehler}</Text>)}
      </ModalDialog>
      <ModalDialog open={frage === "loeschen"}
                   onOpenChange={(o) => !o && setFrage(null)}
                   title={tr("bib.loeschen")}
                   footer={
                     <Button color="red" onClick={() => {
                       void apiSend(`/api/transcripts/${e.id}/delete`,
                                    { confirm: e.id })
                         .then(() => { setFrage(null); onChanged(); })
                         .catch((err) => setDialogFehler(errMsg(err)));
                     }}>{tr("bib.loeschen")}</Button>}>
        <Text size="2">{tr("bib.loeschen.text", { name: e.name })}</Text>
        {dialogFehler && (
          <Text size="1" color="red">{dialogFehler}</Text>)}
      </ModalDialog>
    </>
  );
}
