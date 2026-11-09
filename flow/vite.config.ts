import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // The shared UI component library's unit tests live outside this root
  // (packages/ui-components/test) and run together with flow's own suite.
  test: {
    include: [
      "src/**/*.{test,spec}.{ts,tsx}",
      "../packages/ui-components/test/**/*.{test,spec}.{ts,tsx}",
    ],
  },
  server: {
    port: 5174,
    proxy: {
      "/api": "http://127.0.0.1:3010",
    },
  },
});
