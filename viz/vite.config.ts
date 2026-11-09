/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // The shared UI component library's unit tests live outside this root
  // (packages/ui-components/test) and run together with viz's own suite.
  test: {
    include: [
      "tests/**/*.{test,spec}.{ts,tsx}",
      "src/**/*.{test,spec}.{ts,tsx}",
      "../packages/ui-components/test/**/*.{test,spec}.{ts,tsx}",
    ],
  },
  base: "./",
});
