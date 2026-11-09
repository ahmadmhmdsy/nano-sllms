import type { FlowState } from "../store";
// T3 registry promotion: snapshot types + the static fallback live in
// ../nodes/registry, the port-compat rule in ../nodes/portTypes. They are
// re-exported so existing imports (tests, Inspector, Palette, FlowCanvas,
// api.ts) keep resolving names from the store module.
import { FALLBACK_REGISTRY, DATASET_LABEL_SEMANTIC, CHAT_BUTTON_FEATURE } from "../nodes/registry";
import { portTypesCompatible } from "../nodes/portTypes";
import type { PortSpec, NodeSpec, RegistrySnapshot } from "../nodes/registry";

export { FALLBACK_REGISTRY } from "../nodes/registry";
export { DATASET_LABEL_SEMANTIC, CHAT_BUTTON_FEATURE } from "../nodes/registry";
export { portTypesCompatible } from "../nodes/portTypes";
export type { PortSpec, NodeSpec, RegistrySnapshot } from "../nodes/registry";

/** Editor widget to render for a given registry prop. The train string
 * presets render as selects; numeric_props render as number inputs;
 * anything else is a free-text input. Pure, registry-driven so the
 * per-kind table in the T7 contract is honoured (empty-props kinds lead
 * to the "(no editable properties)" note in the Inspector). */
export type PropWidget = "number" | "select" | "text";

/** The train kind's two string props are preset-name selects. */
const SELECT_PROPS = new Set(["preset", "lr_preset"]);

export function propWidgetType(spec: NodeSpec, prop: string): PropWidget {
  if (SELECT_PROPS.has(prop)) return "select";
  if (spec.numeric_props.includes(prop)) return "number";
  return "text";
}

/** Placeholder select options for the train preset / lr_preset widgets.
 * These names are copied from flow/server/nodes/builtin/train.py
 * PRESETS / LR_PRESETS (which in turn mirror webui/app.py). The inspector dropdowns
 * are a client-side copy on purpose (MVP); actual value parity is enforced
 * by the backend AST-parity test (T5) — a server-side rename that does not
 * update this list shows up as a T5 failure, not a silent config drift. */
export const TRAIN_PRESET_OPTIONS: string[] = [
  "Small (~12M, smoke)",
  "Medium (~101M, pilot)",
  "Large (~226M, target)",
];
export const LR_PRESET_OPTIONS: string[] = [
  "Pretrain 4e-4",
  "Conservative 2e-4",
  "Fine-tune 3e-5",
];

/** Options for a props select widget (empty -> fall back to free text). */
export function propSelectOptions(
  prop: string,
): string[] | null {
  if (prop === "preset") return TRAIN_PRESET_OPTIONS;
  if (prop === "lr_preset") return LR_PRESET_OPTIONS;
  return null;
}

/* ------------------------------------------------------------------ */
/* Registry slice state                                                */
/* ------------------------------------------------------------------ */

import type { StoreApi } from "zustand";
type SetFlow = StoreApi<FlowState>["setState"];
type GetFlow = StoreApi<FlowState>["getState"];

export interface RegistryState {
  /** Registry snapshot; null until GET /api/nodes resolves. */
  registry: RegistrySnapshot | null;
  /** Set when the registry fetch itself failed (fallback list in use). */
  registryError: string | null;
  setRegistry: (r: RegistrySnapshot) => void;
  setRegistryError: (e: string | null) => void;
}

/** Registry snapshot slice of the composed flow store. */
export function createRegistrySlice(set: SetFlow, _get?: GetFlow): RegistryState {
  return {
    registry: null,
    registryError: null,
    setRegistry: (registry) => set({ registry, registryError: null }),
    setRegistryError: (registryError) => set({ registryError }),
  };
}
