// Bibliothek (Ebene 1): Drop → Transkriptions-Job → Eintrag; Liste der
// Transkripte; Import VTT/CSV. Jobs sind flüchtig — die Liste ist die
// Wahrheit (Backend-Bibliothek).
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Badge, Button, Checkbox, Disclosure, EmptyState, ErrorNote, Flex,
  IconButton, Karte, LabeledSelect, ListRow, ModalDialog, Progress,
  Text, TextField,
} from "../components/ui";
import { Icon } from "../components/icons";
import {
  apiGet, apiSend, apiUpload, errMsg, hms, kuerze,
  type EintragMeta, type Job,
  type ModellInfo, type Settings,
} from "../lib/api";
import { jobText, useT } from "../lib/i18n";
import { isTauri, onFileDrop, pickAudio, pickTranskript } from "../lib/tauri";

const AUDIO_EXT = [".mp3", ".m4a", ".aac", ".wav", ".ogg", ".flac",
  ".webm"];

export default function BibliothekModule({ settings, onOpen }: {
  settings: Settings | null;
  onOpen: (id: string) => void;
}) {
  const tr = useT();
  const [eintraege, setEintraege] = useState<EintragMeta[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [fehler, setFehler] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const importRef = useRef<HTMLInputElement>(null);

  const [model, setModel] = useState(settings?.model ?? "large-v3-turbo");
  const [modelle, setModelle] = useState<ModellInfo[]>([]);
  const [language, setLanguage] = useState(settings?.language ?? "de");
  const [diarize, setDiarize] = useState(settings?.diarize ?? true);
  const [range, setRange] = useState(settings?.speaker_range ?? "auto");
  const [threshold, setThreshold] = useState(
    String(settings?.cluster_threshold ?? 0.5));

  const lade = useCallback(async () => {
    try {
      const r = await apiGet<{ transcripts: EintragMeta[] }>(
        "/api/transcripts");
      setEintraege(r.transcripts);
    } catch (e) { setFehler(errMsg(e)); }
  }, []);
  useEffect(() => { void lade(); }, [lade]);
  useEffect(() => {
    void apiGet<{ models: ModellInfo[] }>("/api/models")
      .then((r) => setModelle(r.models)).catch(() => undefined);
  }, []);

  // Job-Poll (1 s) solange etwas läuft; frische Liste bei Abschluss
  const aktiveJobs = jobs.some((j) =>
    !["completed", "failed", "cancelled"].includes(j.status));
  useEffect(() => {
    const t = window.setInterval(() => {
      void apiGet<{ jobs: Job[] }>("/api/jobs").then((r) => {
        setJobs((alt) => {
          const fertigNeu = r.jobs.some((j) => j.status === "completed"
            && alt.find((a) => a.id === j.id)?.status !== "completed");
          if (fertigNeu) void lade();
          return r.jobs;
        });
      }).catch(() => undefined);
    }, aktiveJobs ? 1000 : 5000);
    return () => window.clearInterval(t);
  }, [aktiveJobs, lade]);

  const starteDateien = useCallback(async (pfade: string[]) => {
    setFehler("");
    for (const p of pfade) {
      const ext = p.slice(p.lastIndexOf(".")).toLowerCase();
      if (!AUDIO_EXT.includes(ext)) continue;
      try {
        await apiSend("/api/transcribe-path", {
          path: p, model, language, speaker_range: range,
          cluster_threshold: Number(threshold), diarize });
      } catch (e) { setFehler(errMsg(e)); }
    }
    const r = await apiGet<{ jobs: Job[] }>("/api/jobs");
    setJobs(r.jobs);
  }, [model, language, range, threshold, diarize]);

  // Nativer Tauri-Drop (Pfade — kein Upload großer Audios).
  // EIN Abo, Race-frei (Review-Befund: Re-Subscribe je Options-
  // Änderung verlor Unsubscriber → doppelte Jobs je Drop);
  // die aktuelle Optionen-Fassung kommt über die Ref.
  const starteRef = useRef(starteDateien);
  starteRef.current = starteDateien;
  useEffect(() => {
    let ab: (() => void) | undefined;
    let weg = false;
    void onFileDrop((paths) => { void starteRef.current(paths); })
      .then((f) => { if (weg) f(); else ab = f; });
    return () => { weg = true; ab?.(); };
  }, []);

  const starteUpload = useCallback(async (files: FileList | null) => {
    if (!files?.length) return;
    setFehler("");
    for (const f of Array.from(files)) {
      const fd = new FormData();
      fd.append("file", f);
      fd.append("model", model);
      fd.append("language", language);
      fd.append("speaker_range", range);
      fd.append("cluster_threshold", threshold);
      fd.append("diarize", String(diarize));
      try { await apiUpload("/api/transcribe", fd); }
      catch (e) { setFehler(errMsg(e)); }
    }
    const r = await apiGet<{ jobs: Job[] }>("/api/jobs");
    setJobs(r.jobs);
  }, [model, language, range, threshold, diarize]);

  const importiere = useCallback(async () => {
    setFehler("");
    try {
      if (isTauri()) {
        const p = await pickTranskript();
        if (!p) return;
        await apiSend("/api/import-path", { path: p });
        void lade();
      } else {
        importRef.current?.click();
      }
    } catch (e) { setFehler(errMsg(e)); }
  }, [lade]);

  return (
    <Flex direction="column" gap="3" p="4"
          style={{ height: "100%", overflowY: "auto" }}>
      <div
        onClick={() => {
          if (isTauri()) {
            void pickAudio().then((p) => p && starteDateien(p));
          } else fileRef.current?.click();
        }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault(); setDragOver(false);
          void starteUpload(e.dataTransfer.files);
        }}
        style={{
          border: `2px dashed var(${dragOver ? "--accent-9" : "--gray-a7"})`,
          borderRadius: 12, padding: "28px 16px", textAlign: "center",
          cursor: "pointer",
          background: dragOver ? "var(--accent-a3)" : "var(--gray-a2)",
        }}>
        <Flex direction="column" align="center" gap="1">
          <Icon name="import" size={26} />
          <Text size="3" weight="medium">{tr("bib.drop")}</Text>
          <Text size="1" color="gray">{tr("bib.dropsub")}</Text>
        </Flex>
        <input ref={fileRef} type="file" multiple hidden
               accept={AUDIO_EXT.join(",")}
               onChange={(e) => { void starteUpload(e.target.files);
                 e.target.value = ""; }} />
      </div>

      <Flex gap="3" align="center" wrap="wrap">
        <Button variant="soft" onClick={() => void importiere()}>
          <Icon name="text" /> {tr("bib.import")}</Button>
        <input ref={importRef} type="file" hidden accept=".vtt,.csv"
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

      <Disclosure label={tr("bib.optionen")}>
        <Flex gap="4" wrap="wrap" py="2">
          <LabeledSelect label={tr("bib.modell")} value={model}
            onChange={setModel}
            options={(modelle.length ? modelle.map((m) => m.name)
              : [model]).map((n) => n)} />
          <LabeledSelect label={tr("bib.sprache")} value={language}
            onChange={setLanguage}
            options={["de", "en", "fr", "it", "es", "auto"]} />
          <LabeledSelect label={tr("bib.sprecherzahl")} value={range}
            onChange={setRange}
            options={["auto", "2-2", "2-4", "4-6", "5-8", "6-10"]}
            optionLabels={{ auto: tr("bib.auto") }} />
          <LabeledSelect label={tr("bib.trennung")} value={threshold}
            onChange={setThreshold}
            options={["0.7", "0.5", "0.35", "0.25"]}
            optionLabels={{
              "0.7": tr("bib.trennung.locker"),
              "0.5": tr("bib.trennung.normal"),
              "0.35": tr("bib.trennung.streng"),
              "0.25": tr("bib.trennung.sehr") }} />
          <label style={{ display: "flex", alignItems: "center",
                          gap: 8 }}>
            <Checkbox checked={diarize}
                      onCheckedChange={(v) => setDiarize(v === true)} />
            <Text size="2">{tr("bib.diarize")}</Text>
          </label>
        </Flex>
      </Disclosure>

      {fehler && <ErrorNote>{fehler}</ErrorNote>}

      {jobs.filter((j) => j.status !== "completed").length > 0 && (
        <Karte titel={tr("job.transkribiere")}>
          {jobs.filter((j) => j.status !== "completed").map((j) => (
            <JobZeile key={j.id} job={j} />
          ))}
        </Karte>
      )}

      {eintraege.length === 0
        ? <EmptyState>{tr("bib.leer")}</EmptyState>
        : eintraege.map((e) => (
            <EintragZeile key={e.id} e={e} onOpen={onOpen}
                          onChanged={() => void lade()} />
          ))}
    </Flex>
  );
}

function JobZeile({ job }: { job: Job }) {
  const tr = useT();
  const fertig = ["completed", "failed", "cancelled"]
    .includes(job.status);
  return (
    <Flex direction="column" gap="1" py="2"
          style={{ borderBottom: "1px solid var(--gray-a4)" }}>
      <Flex align="center" gap="2">
        <Text size="2" weight="medium" truncate
              style={{ minWidth: 0 }}>{kuerze(job.filename, 72)}</Text>
        <Badge color={job.status === "failed" ? "red" : "indigo"}>
          {jobText(tr, job.status, job.message)}</Badge>
        <div style={{ flex: 1 }} />
        {!fertig && (
          <Button size="1" variant="soft" color="red"
                  onClick={() => void apiSend(
                    `/api/jobs/${job.id}/cancel`)}>
            {tr("bib.abbrechen")}</Button>
        )}
      </Flex>
      {!fertig && <Progress value={job.progress} />}
      {job.status === "failed" && (
        <Text size="1" color="red">
          {tr("bib.jobfehler", { e: job.error ?? "?" })}</Text>
      )}
      {!fertig && job.partial_text && (
        <Text size="1" color="gray" style={{
          maxHeight: 60, overflow: "hidden", fontStyle: "normal" }}>
          {job.partial_text.slice(-300)}</Text>
      )}
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
