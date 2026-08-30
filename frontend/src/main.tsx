import React from "react";
import ReactDOM from "react-dom/client";
import { Theme } from "@radix-ui/themes";
import "@radix-ui/themes/styles.css";
import App from "./App";
import "./styles.css";

/** Statt weißem Fenster: Fehlertext sichtbar (enrich-Lehre — im
    WKWebView gibt es keinen Inspector). */
class Fehlerfang extends React.Component<
  { children: React.ReactNode }, { fehler: string | null }> {
  state = { fehler: null as string | null };
  static getDerivedStateFromError(e: unknown) {
    return { fehler: e instanceof Error
      ? `${e.message}\n${e.stack ?? ""}` : String(e) };
  }
  render() {
    if (this.state.fehler) {
      return <pre style={{ padding: 24, whiteSpace: "pre-wrap",
                           fontSize: 12 }}>{this.state.fehler}</pre>;
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Theme accentColor="indigo" grayColor="slate" radius="medium"
           style={{ minHeight: "100vh" }}>
      <Fehlerfang><App /></Fehlerfang>
    </Theme>
  </React.StrictMode>,
);
