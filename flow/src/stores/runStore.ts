import { api, errorMessage } from "../api";
import { flowDocument, isValidFlowName } from "./graphStore";
import type { FlowState } from "../store";

/** UI-level run state; setRunStatus stays unwired until the Task 9/10 run
 * wiring. The raw backend payload type is ApiRunStatus in api.ts
 * (GET /api/run/status contract shape). */
export interface RunStatus {
  state: "idle" | "running" | "done" | "error";
  message?: string;
  /** Backend exit_code (parallel to state done/error; null while running).
   * Shared with the PipelineNode dot tooltip ("exit code N"). */
  exitCode?: number | null;
}

/** Poll interval math for the Run log watcher (carry-over T9 finding (d)):
 * the base cadence is 2 s; after 3 consecutive poll failures stretch to
 * ~30 s so a dead backend does not spam fetch, and the successful read
 * resets the counter (failure count is kept by the watcher loop). */
export const POLL_BASE_MS = 2000;
export const POLL_BACKOFF_MS = 30000;
export const POLL_FAILS_BEFORE_BACKOFF = 3;

export function nextPollDelay(consecutiveFailures: number): number {
  return consecutiveFailures >= POLL_FAILS_BEFORE_BACKOFF
    ? POLL_BACKOFF_MS
    : POLL_BASE_MS;
}

/* ------------------------------------------------------------------ */
/* Run slice state                                                     */
/* ------------------------------------------------------------------ */

import type { StoreApi } from "zustand";
type SetFlow = StoreApi<FlowState>["setState"];
type GetFlow = StoreApi<FlowState>["getState"];

export interface RunState {
  runStatus: RunStatus | null;
  /** Confirmation line from the last successful Stop (STOP flag path). */
  stopInfo: string | null;
  setRunStatus: (s: RunStatus | null) => void;
  /** Task 10 wiring: arm a run (PUT-save then POST /api/run -> runStatus
   * running + Run log tab), and request the checkpoint-aligned stop
   * (POST /api/run/stop). */
  runGraph: () => Promise<boolean>;
  stopRun: () => Promise<boolean>;
}

/** Run slice of the composed flow store. */
export function createRunSlice(set: SetFlow, get: GetFlow): RunState {
  return {
    runStatus: null,
    stopInfo: null,
    setRunStatus: (runStatus) => set({ runStatus }),
    // Run = PUT-save the current graph under its slug, then POST /api/run.
    // 400 preflight errors[] and 409 busy detail land in the toast channel
    // verbatim (formatApiError already prefers the errors/detail shapes);
    // on success runStatus arms the Run log watcher (which start polls).
    runGraph: async () => {
      const name = get().currentFlowName;
      if (name === null || !isValidFlowName(name)) {
        set({ error: "Run needs a saved flow: use Save first (name must match [a-z0-9-]{1,64})" });
        return false;
      }
      try {
        await api.saveFlow(name, flowDocument(name, get().graph));
      } catch (e) {
        set({ error: errorMessage(e) });
        return false;
      }
      try {
        const r = await api.runFlow(name);
        set({
          runStatus: { state: "running", message: "started pid " + String(r.pid), exitCode: undefined },
          stopInfo: null,
          inspectorTab: "run-log",
          error: null,
        });
        return true;
      } catch (e) {
        set({ error: errorMessage(e) });
        return false;
      }
    },
    // Stop is the ONLY run control by design (NO kill button): it writes
    // the U11 STOP flag; the trainer exits cleanly at the next checkpoint
    // save. A 409 ("no live GPU job") surface through the toast verbatim.
    stopRun: async () => {
      try {
        const r = await api.runStop();
        set({ error: null, stopInfo: "stop flag written: " + r.stop_flag });
        return true;
      } catch (e) {
        set({ error: errorMessage(e) });
        return false;
      }
    },
  };
}
