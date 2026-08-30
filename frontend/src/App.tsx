// App-Rahmen: Boot-Gate (Tauri startet das Backend), First-Run
// (Speicherort), dann Bibliothek ⇄ Editor (Drilldown, enrich-
// Werkstatt-Muster) + Einstellungen.
import { useCallback, useEffect, useState } from "react";
import { Badge, Button, Flex, Heading, SegTabs, Text }
  from "./components/ui";
import { Icon } from "./components/icons";
import { apiGet, apiSend, errMsg, type Settings } from "./lib/api";
import { setSprache, useT, type Sprache } from "./lib/i18n";
import { backendStarten, isTauri, pickOrdner } from "./lib/tauri";
import AiTranscriptModule from "./modules/AiTranscriptModule";
import EditorModule from "./modules/EditorModule";
import EinstellungenModule from "./modules/EinstellungenModule";
import HumanEditorModule from "./modules/HumanEditorModule";

type Boot = "lade" | "bereit" | "fehler";

export default function App() {
  const tr = useT();
  const [boot, setBoot] = useState<Boot>("lade");
  const [bootFehler, setBootFehler] = useState("");
  const [settings, setSettings] = useState<Settings | null>(null);
  // Drei Tabs (User 2026-08-30): AI-Transcript (Default) |
  // Human-Editor (Bibliotheks-Spiegel + Import, Editor-Drilldown) |
  // Einstellungen
  const [tab, setTab] = useState<"ai" | "editor" | "einstellungen">(
    "ai");
  const [editorId, setEditorId] = useState<string | null>(null);

  const starte = useCallback(async () => {
    setBoot("lade");
    try {
      if (isTauri()) await backendStarten();
      const s = await apiGet<Settings>("/api/settings");
      setSettings(s);
      if (s.ui_language) setSprache(s.ui_language as Sprache);
      setBoot("bereit");
    } catch (e) {
      setBootFehler(errMsg(e));
      setBoot("fehler");
    }
  }, []);
  useEffect(() => { void starte(); }, [starte]);

  if (boot !== "bereit") {
    return (
      <Flex align="center" justify="center" direction="column" gap="3"
            style={{ height: "100vh" }}>
        <Heading size="5">{tr("app.titel")}</Heading>
        <Text size="2" color="gray">{tr("app.untertitel")}</Text>
        {boot === "lade"
          ? <Text size="2">{tr("app.boot")}</Text>
          : <>
              <Badge color="red">{tr("app.bootfehler")}</Badge>
              <Text size="1" color="gray"
                    style={{ maxWidth: 480, whiteSpace: "pre-wrap" }}>
                {bootFehler}</Text>
              <Button onClick={() => void starte()}>
                {tr("app.nochmal")}</Button>
            </>}
      </Flex>
    );
  }

  if (settings && !settings.library_root) {
    return <FirstRun onDone={(s) => setSettings(s)} />;
  }

  return (
    <Flex direction="column" style={{ height: "100vh" }}>
      <Flex align="center" gap="3" px="4" py="2"
            style={{ borderBottom: "1px solid var(--gray-a5)" }}>
        <Heading size="4">{tr("app.titel")}</Heading>
        <div style={{ flex: 1 }} />
        {/* im Editor-Drilldown ist KEIN Tab aktiv — so feuert der
            Klick auf „Human-Editor" ein onChange und verlässt den
            Editor zur Liste (Review-Befund) */}
        <SegTabs value={editorId ? "" : tab}
                 onChange={(v) => { setEditorId(null);
                   setTab(v as "ai" | "editor" | "einstellungen"); }}
                 options={[
                   { value: "ai", label: tr("tab.ai"),
                     icon: "sample" },
                   { value: "editor", label: tr("tab.editor"),
                     icon: "edit" },
                   { value: "einstellungen",
                     label: tr("tab.einstellungen"),
                     icon: "settings" }]} />
      </Flex>
      <div style={{ flex: 1, minHeight: 0 }}>
        {editorId
          ? <EditorModule id={editorId}
                          onExit={() => setEditorId(null)} />
          : tab === "ai"
            ? <AiTranscriptModule settings={settings}
                onEdit={(id) => { setTab("editor");
                  setEditorId(id); }} />
            : tab === "editor"
              ? <HumanEditorModule
                  onOpen={(id) => setEditorId(id)} />
              : <EinstellungenModule settings={settings}
                                     onChange={setSettings} />}
      </div>
    </Flex>
  );
}

function FirstRun({ onDone }: { onDone: (s: Settings) => void }) {
  const tr = useT();
  const [fehler, setFehler] = useState("");
  const setze = async (root: string) => {
    try {
      onDone(await apiSend<Settings>("/api/settings",
                                     { library_root: root }));
    } catch (e) { setFehler(errMsg(e)); }
  };
  return (
    <Flex align="center" justify="center" direction="column" gap="4"
          style={{ height: "100vh" }}>
      <Icon name="folder" size={40} />
      <Heading size="5">{tr("firstrun.titel")}</Heading>
      <Text size="2" color="gray"
            style={{ maxWidth: 440, textAlign: "center" }}>
        {tr("firstrun.text")}</Text>
      <Flex gap="3">
        <Button onClick={() => void setze("default")}>
          {tr("firstrun.standard")}</Button>
        {isTauri() && (
          <Button variant="soft" onClick={() => {
            void pickOrdner().then((p) => { if (p) void setze(p); });
          }}>{tr("firstrun.waehlen")}</Button>
        )}
      </Flex>
      {fehler && <Text size="1" color="red">{fehler}</Text>}
    </Flex>
  );
}
