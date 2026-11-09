import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./global.css";

// visible error surface (dev habit from the static build — plan §5)
function jsBanner(kind: string, msg: string) {
  const d = document.createElement("div");
  d.dataset.jserr = kind;
  d.style.cssText = "position:fixed;top:0;left:0;right:0;background:#e8635a;color:#fff;font:11px monospace;padding:6px;z-index:99";
  d.textContent = kind + ": " + msg;
  document.body.appendChild(d);
}
window.addEventListener("error", (e) => jsBanner("JS-ERROR", e.message));
window.addEventListener("unhandledrejection", (e) => jsBanner("PERROR", String(e.reason)));

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
