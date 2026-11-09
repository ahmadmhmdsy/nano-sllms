/**
 * Undo/redo history stack for the pipeline graph (M1.1).
 *
 * A command-pattern history: each mutating action pushes the PRE-mutation
 * graph onto the past-stack before applying the change. undo() pops the
 * past-stack, saves the current graph to the future-stack, and restores.
 * redo() is the inverse. A new mutation clears the future-stack (standard
 * editor semantics: you cannot redo after a new action).
 *
 * Only "commit" actions push history (add, connect, delete, props/label
 * edit, setGraph). Transient changes (position drags, selection) do NOT
 * push history — otherwise every mouse-move during a drag would create
 * an undo step.
 *
 * The history is capped at HISTORY_LIMIT to prevent unbounded memory use
 * on large graphs.
 */

import type { Graph } from "./graphStore";

const HISTORY_LIMIT = 100;

export interface HistoryEntry {
  /** The graph state BEFORE the mutation. */
  before: Graph;
  /** Human-readable description of the action (for UI tooltips). */
  label: string;
}

export interface HistoryState {
  past: HistoryEntry[];
  future: HistoryEntry[];
  /** Push a new history entry. Clears the future-stack (new action = no redo). */
  push: (before: Graph, label: string) => void;
  /** Restore the previous state. Returns the graph to restore, or null. */
  undo: () => { graph: Graph; label: string } | null;
  /** Re-apply the next state. Returns the graph to restore, or null. */
  redo: () => { graph: Graph; label: string } | null;
  /** Clear all history (e.g., on flow open). */
  clear: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
}

import type { StoreApi } from "zustand";
import type { FlowState } from "../store";

type SetFlow = StoreApi<FlowState>["setState"];
type GetFlow = StoreApi<FlowState>["getState"];

export function createHistorySlice(set: SetFlow, get: GetFlow): HistoryState {
  return {
    past: [],
    future: [],

    push: (before, label) => {
      const { past } = get();
      const next = [...past, { before, label }];
      // Cap the history: drop oldest entries if over the limit
      if (next.length > HISTORY_LIMIT) next.splice(0, next.length - HISTORY_LIMIT);
      set({ past: next, future: [] });
    },

    undo: () => {
      const { past, future, graph } = get();
      if (past.length === 0) return null;
      const entry = past[past.length - 1];
      const newPast = past.slice(0, -1);
      const newFuture = [...future, { before: graph, label: entry.label }];
      set({
        past: newPast,
        future: newFuture,
        graph: entry.before,
        projection: [],
        validatedDoc: null,
      });
      return { graph: entry.before, label: entry.label };
    },

    redo: () => {
      const { past, future, graph } = get();
      if (future.length === 0) return null;
      const entry = future[future.length - 1];
      const newFuture = future.slice(0, -1);
      const newPast = [...past, { before: graph, label: entry.label }];
      set({
        past: newPast,
        future: newFuture,
        graph: entry.before,
        projection: [],
        validatedDoc: null,
      });
      return { graph: entry.before, label: entry.label };
    },

    clear: () => set({ past: [], future: [] }),

    canUndo: () => get().past.length > 0,
    canRedo: () => get().future.length > 0,
  };
}
