/**
 * Task 11 — infer node render-only shortcut + handoff dialog.
 *
 * Honest MVP: no server endpoint. Clicking "Open in webui Chat" always
 * (1) opens the store-driven dialog with the manual webui launch
 * instructions and (2) attempts window.open(WEBUI_URL, "_blank"). If the
 * popup is blocked or the webui is not running, the dialog is the
 * fallback the user still sees; the new tab may show a browser error —
 * that is the documented, acceptable MVP behavior (brief, Orchestrator
 * addition).
 */

/** Default webui URL. Read (not invented) from webui/app.py:
 * CHAT_PORT = 7860 (app.py line 35) and the launch block uses
 * server_name="127.0.0.1" with --port default CHAT_PORT (lines 1534,
 * 1551-1552). Kept as ONE clearly-named module constant on purpose. */
export const WEBUI_URL = "http://127.0.0.1:7860";

/** Instruction shown on the infer node body (honest, no fabricated
 * path): the webui-recognized checkpoint path is not determinable
 * client-side — the ckpt-dir value arrives as the connected edge from
 * the upstream node, not as node props. */
export const INFER_CHECKPOINT_NOTE =
  "checkpoint: see connected input node's output";

export interface InferHandoffDialogInfo {
  url: string;
  /** venv command line to start webui/app.py by hand (AGENTS.md §3
   * venv rule; every Python run goes through the project venv). */
    launchCommand: "& ./.venv/Scripts/python.exe webui/app.py",
  checkpointNote: string;
  /** Tab label the user opens after launching the webui. */
  chatTab: string;
}

/** Pure builder for the dialog content (exported for direct unit
 * testing — no component render needed). */
export function inferHandoffDialog(): InferHandoffDialogInfo {
  return {
    url: WEBUI_URL,
    launchCommand: "& ./.venv/Scripts/python.exe webui/app.py",
    checkpointNote: INFER_CHECKPOINT_NOTE,
    chatTab: "Chat",
  };
}

/** Best-effort open of the default webui location in a new tab via
 * window.open(WEBUI_URL, "_blank"). The result is window.open's return
 * (null when the popup is blocked); the dialog is ALWAYS shown
 * regardless. Exported so the click path is unit-testable via
 * vi.spyOn(window, "open"). */
export function attemptWebuiOpen(win: Window = window): Window | null {
  return win.open(WEBUI_URL, "_blank");
}
