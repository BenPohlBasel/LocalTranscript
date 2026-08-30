import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Port 1421 fest: darauf zeigt Tauris devUrl. Der /api-Proxy gilt nur im
// Browser-Dev; das Tauri-Fenster spricht 127.0.0.1:44100 direkt (CORS).
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1421,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:44100" },
  },
});
