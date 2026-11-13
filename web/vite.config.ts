import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The production build goes straight into the Python package so the Hub serves it.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/badamoa/hub/static", emptyOutDir: true },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8470",
      "/install": "http://127.0.0.1:8470",
      "/mcp": "http://127.0.0.1:8470",
    },
  },
});
