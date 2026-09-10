// Einstellungen: Speicherort, Standard-Optionen, UI-Sprache, Lizenzen
// (inkl. Recursive-OFL-Nennung — Pflicht, Fonts liegen im Bundle).
import { useEffect, useState } from "react";
import {
  Button, Flex, Grid, Karte, LabeledSelect, Text, TextField,
} from "../components/ui";
import {
  apiGet, apiSend, errMsg, type ModellInfo, type Settings,
} from "../lib/api";
import { setSprache, useT, type Sprache } from "../lib/i18n";
import { isTauri, ordnerOeffnen, pickOrdner } from "../lib/tauri";

const BIAS_URL = "https://bias.city/"
  + "b-ias-basel-institut-fuer-angewandte-stadtforschung/";

export default function EinstellungenModule({ settings, onChange }: {
  settings: Settings | null;
  onChange: (s: Settings) => void;
}) {
  const tr = useT();
  const [fehler, setFehler] = useState("");
  const [modelle, setModelle] = useState<ModellInfo[]>([]);
  const [modelsDir, setModelsDir] = useState("");
  useEffect(() => {
    void apiGet<{ models: ModellInfo[]; models_dir: string }>(
      "/api/models").then((r) => {
        setModelle(r.models); setModelsDir(r.models_dir);
      }).catch(() => undefined);
  }, []);

  const setze = (aend: Record<string, unknown>) => {
    void apiSend<Settings>("/api/settings", aend)
      .then(onChange).catch((e) => setFehler(errMsg(e)));
  };
  // E-Mail: lokal tippen, beim Verlassen des Feldes oder Enter sichern
  const [mail, setMail] = useState(settings?.user_email ?? "");
  useEffect(() => { setMail(settings?.user_email ?? ""); },
            [settings?.user_email]);
  if (!settings) return null;

  return (
    // Karten-Raster wie enrich-Einstellungen (User 2026-08-30):
    // responsives 1–2-Spalten-Grid, volle Breite
    <Grid columns={{ initial: "1", md: "2" }} gap="4" p="4"
          style={{ height: "100%", overflowY: "auto",
                   alignContent: "start" }}>
      <Karte titel={tr("st.speicherort")}
             subline={tr("st.speicherort.text")}>
        <Flex align="center" gap="2">
          <Text size="2" style={{ flex: 1, wordBreak: "break-all" }}>
            {settings.library_root}</Text>
          {isTauri() && (
            <>
              <Button size="1" variant="soft" onClick={() => {
                void pickOrdner().then((p) => p
                  && setze({ library_root: p }));
              }}>{tr("st.aendern")}</Button>
              <Button size="1" variant="ghost" onClick={() =>
                void ordnerOeffnen(settings.library_root)}>
                {tr("bib.ordner")}</Button>
            </>
          )}
        </Flex>
      </Karte>

      <Karte titel={tr("st.standards")}>
        <Flex gap="4" wrap="wrap">
          <LabeledSelect label={tr("bib.modell")}
            value={settings.model}
            onChange={(v) => setze({ model: v })}
            options={(modelle.length ? modelle.map((m) => m.name)
              : [settings.model])} />
          <LabeledSelect label={tr("bib.sprache")}
            value={settings.language}
            onChange={(v) => setze({ language: v })}
            options={["de", "en", "fr", "it", "es", "auto"]} />
          {/* Gleichzeitige Läufe (User 2026-09-09). Standard 1: mehr
              als einer teilt sich dieselbe GPU — vier parallel sind
              nicht schneller als vier nacheinander, nur unübersichtlich. */}
          <LabeledSelect label={tr("st.parallel")}
            value={String(settings.max_parallel ?? 1)}
            onChange={(v) => setze({ max_parallel: Number(v) })}
            options={["1", "2", "3", "4"]}
            optionLabels={{ "1": tr("st.parallel.eins") }} />
          <LabeledSelect label={tr("st.uisprache")}
            value={settings.ui_language}
            onChange={(v) => { setSprache(v as Sprache);
              setze({ ui_language: v }); }}
            options={["de", "en", "fr", "it"]}
            optionLabels={{ de: "Deutsch", en: "English",
              fr: "Français", it: "Italiano" }} />
        </Flex>
        {modelsDir && (
          <Text size="1" color="gray" mt="2" as="div">
            {tr("st.modelle", { d: modelsDir })}</Text>
        )}
      </Karte>

      <Karte titel={tr("st.identitaet")} subline={tr("st.identitaet.sub")}>
        <Flex direction="column" gap="3">
          <Flex direction="column" gap="1">
            <Text size="1" weight="medium">{tr("st.email")}</Text>
            <TextField.Root size="2" type="email" value={mail}
              placeholder="name@institut.ch"
              onChange={(e) => setMail(e.target.value)}
              onBlur={() => { if (mail.trim() !== settings.user_email)
                setze({ user_email: mail.trim() }); }}
              onKeyDown={(e) => { if (e.key === "Enter")
                (e.target as HTMLInputElement).blur(); }} />
            <Text size="1" color="gray">{tr("st.email.hinweis")}</Text>
          </Flex>
          <Flex direction="column" gap="1">
            <Text size="1" weight="medium">{tr("st.install")}</Text>
            <Flex gap="2" align="center" wrap="wrap">
              <Text size="1" style={{ fontFamily: "monospace" }}>
                {settings.install_id}</Text>
              <Button size="1" variant="soft"
                      onClick={() => setze({ install_id: "neu" })}>
                {tr("st.install.neu")}</Button>
            </Flex>
            <Text size="1" color="gray">{tr("st.install.hinweis")}</Text>
          </Flex>
        </Flex>
      </Karte>

      <Karte titel={tr("st.datenschutz")}>
        <Text size="1" color="gray">{tr("st.datenschutz.text")}</Text>
      </Karte>

      <Karte titel={tr("st.lizenzen")}>
        <Flex direction="column" gap="2">
          <Text size="1" color="gray">{tr("st.lizenzen.text")}</Text>
          <Text size="1" color="gray">{tr("st.agpl")}</Text>
          <Flex gap="2" wrap="wrap">
            <Button size="1" variant="soft" onClick={() =>
              void ordnerOeffnen(
                "https://github.com/BenPohlBasel/LocalTranscript")}>
              {tr("st.link.repo")}</Button>
            <Button size="1" variant="soft" onClick={() =>
              void ordnerOeffnen("https://github.com/BenPohlBasel/"
                + "LocalTranscript/releases")}>
              {tr("st.link.releases")}</Button>
            <Button size="1" variant="soft" onClick={() =>
              void ordnerOeffnen("https://ffmpeg.martin-riedl.de")}>
              {tr("st.link.ffmpegbuild")}</Button>
            <Button size="1" variant="soft" onClick={() =>
              void ordnerOeffnen("https://ffmpeg.org/download.html")}>
              {tr("st.link.ffmpegsrc")}</Button>
          </Flex>
        </Flex>
      </Karte>

      <Karte titel={tr("st.app")}
             subline={tr("st.app.sub")}>
        <Flex direction="column" gap="2">
          <Text size="1" color="gray">{tr("st.app.text")}</Text>
          <Flex gap="2">
            <Button size="1" variant="soft" onClick={() =>
              void ordnerOeffnen(
                "https://github.com/BenPohlBasel/LocalTranscript")}>
              GitHub</Button>
          </Flex>
        </Flex>
      </Karte>

      {/* Eigene Karte für die Austauschformate (User 2026-09-09):
          die Spezifikations-Lizenzen sind eine andere Frage als die
          der mitgelieferten Werkzeuge. */}
      <Karte titel={tr("st.formate")} subline={tr("st.formate.sub")}>
        <Flex direction="column" gap="2">
          <Text size="1" color="gray">{tr("st.formate.refi")}</Text>
          <Text size="1" color="gray">{tr("st.formate.enrich")}</Text>
          <Text size="1" color="gray">{tr("st.formate.xsd")}</Text>
          <Flex gap="2" wrap="wrap">
            <Button size="1" variant="soft" onClick={() =>
              void ordnerOeffnen("https://www.qdasoftware.org/")}>
              {tr("st.link.refi")}</Button>
          </Flex>
        </Flex>
      </Karte>

      <Karte titel={tr("st.bias")} subline={tr("st.bias.sub")}>
        <Flex direction="column" gap="2">
          <Text size="1" color="gray">{tr("st.bias.text")}</Text>
          <Flex gap="2">
            <Button size="1" variant="soft" onClick={() =>
              void ordnerOeffnen(BIAS_URL)}>
              {tr("st.bias.link")}</Button>
          </Flex>
        </Flex>
      </Karte>

      {fehler && <Text size="1" color="red">{fehler}</Text>}
    </Grid>
  );
}
