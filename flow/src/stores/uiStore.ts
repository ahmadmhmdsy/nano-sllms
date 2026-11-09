import type { FlowState } from "../store";
import { searchMatches, nextMatchIndex } from "./search";

/** Toast-friendly truncation of a long validation error list: keep the
 * first 3 lines verbatim and summarize the rest as "+N more" (the toast
 * has a max-width and the backend can emit a line per validator check). */
export function truncateErrorList(errors: string[], keep = 3): string[] {
  if (errors.length <= keep) return errors;
  return [...errors.slice(0, keep), "+" + (errors.length - keep) + " more"];
}

export type InspectorTab = "properties" | "run-log" | "preview";

/** Task M2.4: a pending destructive-action confirmation (Clear canvas /
 * Delete flow). The request lives in the store — like
 * inferHandoffNodeId — so the Toolbar wiring is unit-testable through
 * useFlowStore.getState() without a DOM testing library. */
export interface ConfirmRequest {
  title: string;
  message: string;
  /** The destructive effect; run only after the user confirms. */
  action: () => void;
}

/* ------------------------------------------------------------------ */
/* UI slice state                                                      */
/* ------------------------------------------------------------------ */

import type { StoreApi } from "zustand";
type SetFlow = StoreApi<FlowState>["setState"];
type GetFlow = StoreApi<FlowState>["getState"];

/** One center request FlowCanvas answers with setCenter (only available
 * * inside the ReactFlowProvider, so the Toolbar's Enter key routes
 * through the store). The position is the node's flow coordinates,
 * snapshotted at request time. */
export interface CenterRequest {
  nodeId: string;
  x: number;
  y: number;
  /** Monotonic per request: FlowCanvas answers each request exactly once
   * even under StrictMode's double-invoked effects. */
  nonce: number;
}

export interface UiState {
  selection: string | null;
  inspectorTab: InspectorTab;
  paletteCollapsed: boolean;
  inspectorCollapsed: boolean;
  /** User-visible error / toast line; reducers never mutate graph on reject. */
  error: string | null;
  /** Task 11 infer-handoff dialog state. */
  inferHandoffNodeId: string | null;
  /** Task M2.4: pending destructive-action confirmation; null = closed. */
  confirm: ConfirmRequest | null;
  openConfirm: (title: string, message: string, action: () => void) => void;
  dismissConfirm: () => void;
  /** Task M2.2 node search: toolbar filter text; "" = no search active. */
  searchQuery: string;
  /** Task M2.2: latest center request (search Enter cycling). */
  centerRequest: CenterRequest | null;
  setSelection: (id: string | null) => void;
  setInspectorTab: (t: InspectorTab) => void;
  setPaletteCollapsed: (c: boolean) => void;
  setInspectorCollapsed: (c: boolean) => void;
  setError: (e: string | null) => void;
  /** Task 11 infer-handoff dialog slice (pure, reducer-safe; see
   * src/inferHandoff.ts for the rendered content). */
  openInferHandoff: (nodeId: string) => void;
  dismissInferHandoff: () => void;
  /** Task M2.2: update the toolbar search query. */
  setSearchQuery: (q: string) => void;
  /** Task M2.2: Enter in the search box - select + center on the next
   * match (pure searchMatches/nextMatchIndex; no-match is a no-op). */
  cycleSearchMatch: () => void;
}

/** UI slice of the composed flow store: selection, inspector tabs,
 * collapse flags, toast channel, dialog state, node search. */
export function createUiSlice(set: SetFlow, get: GetFlow): UiState {
  // Monotonic center-request nonce (per store instance; uniqueness is all
  // FlowCanvas needs to dedupe effect re-runs).
  let centerNonce = 0;
  return {
    selection: null,
    inspectorTab: "properties",
    paletteCollapsed: false,
    inspectorCollapsed: false,
    error: null,
    inferHandoffNodeId: null,
    confirm: null,
    searchQuery: "",
    centerRequest: null,

    setSelection: (selection) => set({ selection }),
    setInspectorTab: (inspectorTab) => set({ inspectorTab }),
    setPaletteCollapsed: (paletteCollapsed) => set({ paletteCollapsed }),
    setInspectorCollapsed: (inspectorCollapsed) => set({ inspectorCollapsed }),
    setError: (error) => set({ error }),
    openInferHandoff: (nodeId) => set({ inferHandoffNodeId: nodeId }),
    dismissInferHandoff: () => set({ inferHandoffNodeId: null }),
    openConfirm: (title, message, action) => set({ confirm: { title, message, action } }),
    dismissConfirm: () => set({ confirm: null }),
    setSearchQuery: (searchQuery) => set({ searchQuery }),
    cycleSearchMatch: () => {
      const g = get();
      const matches = searchMatches(g.graph, g.searchQuery);
      if (matches.length === 0) return;
      const id = matches[nextMatchIndex(matches, g.selection)];
      const node = g.graph.nodes.find((n) => n.id === id);
      if (!node) return;
      centerNonce += 1;
      set({
        selection: id,
        centerRequest: { nodeId: id, x: node.position.x, y: node.position.y, nonce: centerNonce },
      });
    },
  };
}
