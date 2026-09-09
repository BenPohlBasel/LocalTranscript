import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Port 1421 fest: darauf zeigt Tauris devUrl. Der /api-Proxy gilt nur im
// Browser-Dev; das Tauri-Fenster spricht 127.0.0.1:5628 direkt (CORS).
// LT_SERVE_PORT zieht auch hier — so lässt sich der Dev gegen ein
// Scratch-Backend fahren, ohne an einem laufenden vorbeizugreifen.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1421,
    strictPort: true,
    proxy: { "/api": `http://127.0.0.1:${process.env.LT_SERVE_PORT || 5628}` },
  },
});
