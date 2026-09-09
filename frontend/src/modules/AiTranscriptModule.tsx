// AI-Transcript (Default-Tab, User 2026-08-30): Dropzone + offene
// Optionen + BATCH-Liste der Läufe (auch fertige, mit Sprung in den
// Human-Editor). Die Bibliotheks-Liste lebt im Human-Editor-Tab.
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Badge, Button, Checkbox, Disclosure, ErrorNote, Flex, Karte,
  LabeledSelect, Progress, Text,
} from "../components/ui";
import { Icon } from "../components/icons";
import {
  apiGet, apiSend, apiUpload, errMsg, kuerze, type Job,
  type ModellInfo, type Settings,
} from "../lib/api";
import { jobText, useT } from "../lib/i18n";
import { isTauri, onFileDrop, pickAudio } from "../lib/tauri";

const AUDIO_EXT = [".mp3", ".m4a", ".aac", ".wav", ".ogg", ".flac",
  ".webm"];

export default function AiTranscriptModule({ settings, onEdit }: {
  settings: Settings | null;
  onEdit: (id: string) => void;
}) {
  const tr = useT();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [fehler, setFehler] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const [model, setModel] = useState(settings?.model ?? "large-v3-turbo");
  const [modelle, setModelle] = useState<ModellInfo[]>([]);
  const [language, setLanguage] = useState(settings?.language ?? "de");
  const [diarize, setDiarize] = useState(settings?.diarize ?? true);
  const [range, setRange] = useState(settings?.speaker_range ?? "auto");
  const [threshold, setThreshold] = useState(
    String(settings?.cluster_threshold ?? 0.5));

  useEffect(() => {
    void apiGet<{ models: ModellInfo[] }>("/api/models")
      .then((r) => setModelle(r.models)).catch(() => undefined);
    void apiGet<{ jobs: Job[] }>("/api/jobs")
      .then((r) => setJobs(r.jobs)).catch(() => undefined);
  }, []);

  const aktiveJobs = jobs.some((j) =>
    !["completed", "failed", "cancelled"].includes(j.status));
  useEffect(() => {
    const t = window.setInterval(() => {
      void apiGet<{ jobs: Job[] }>("/api/jobs")
        .then((r) => setJobs(r.jobs)).catch(() => undefined);
    }, aktiveJobs ? 1000 : 5000);
    return () => window.clearInterval(t);
  }, [aktiveJobs]);

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
          <Icon name="import" size={24} />
          <Text size="3" weight="medium">{tr("bib.drop")}</Text>
          <Text size="1" color="gray">{tr("bib.dropsub")}</Text>
        </Flex>
        <input ref={fileRef} type="file" multiple hidden
               accept={AUDIO_EXT.join(",")}
               onChange={(e) => { void starteUpload(e.target.files);
                 e.target.value = ""; }} />
      </div>

      <Disclosure label={tr("bib.optionen")} defaultOpen>
        <Flex gap="4" wrap="wrap" py="2">
          <LabeledSelect label={tr("bib.modell")} value={model}
            onChange={setModel}
            options={(modelle.length ? modelle.map((m) => m.name)
              : [model])} />
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

      {jobs.length > 0 && (
        <Karte titel={tr("ai.batch")}>
          {jobs.map((j) => (
            <JobZeile key={j.id} job={j} onEdit={onEdit} />
          ))}
        </Karte>
      )}
    </Flex>
  );
}

function JobZeile({ job, onEdit }: {
  job: Job; onEdit: (id: string) => void;
}) {
  const tr = useT();
  const fertig = ["completed", "failed", "cancelled"]
    .includes(job.status);
  return (
    <Flex direction="column" gap="1" py="2"
          style={{ borderBottom: "1px solid var(--gray-a4)" }}>
      <Flex align="center" gap="2">
        <Text size="2" weight="medium" truncate
              style={{ minWidth: 0 }}>{kuerze(job.filename, 72)}</Text>
        <Badge color={job.status === "failed" ? "red"
          : job.status === "completed" ? "green" : "indigo"}>
          {jobText(tr, job.status, job.message)}</Badge>
        <div style={{ flex: 1 }} />
        {job.status === "completed" && job.eintrag && (
          <Button size="1" variant="soft"
                  onClick={() => onEdit(job.eintrag!)}>
            <Icon name="edit" size={14} /> {tr("allg.bearbeiten")}
          </Button>
        )}
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
          maxHeight: 60, overflow: "hidden" }}>
          {job.partial_text.slice(-300)}</Text>
      )}
    </Flex>
  );
}
