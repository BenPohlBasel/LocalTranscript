// Einstellungen: Speicherort, Standard-Optionen, UI-Sprache, Lizenzen
// (inkl. Recursive-OFL-Nennung — Pflicht, Fonts liegen im Bundle).
import { useEffect, useState } from "react";
import {
  Button, Flex, Karte, LabeledSelect, Text,
} from "../components/ui";
import {
  apiGet, apiSend, errMsg, type ModellInfo, type Settings,
} from "../lib/api";
import { setSprache, useT, type Sprache } from "../lib/i18n";
import { isTauri, ordnerOeffnen, pickOrdner } from "../lib/tauri";

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
  if (!settings) return null;

  return (
    <Flex direction="column" gap="3" p="4"
          style={{ height: "100%", overflowY: "auto", maxWidth: 720 }}>
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

      <Karte titel={tr("st.datenschutz")}>
        <Text size="1" color="gray">{tr("st.datenschutz.text")}</Text>
      </Karte>

      <Karte titel={tr("st.lizenzen")}>
        <Text size="1" color="gray">{tr("st.lizenzen.text")}</Text>
      </Karte>

      {fehler && <Text size="1" color="red">{fehler}</Text>}
    </Flex>
  );
}
