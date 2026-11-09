/**
 * Registry snapshot types + static fallback registry (T3 of
 * docs/plans/flow-registry-promotion.md).
 *
 * The SERVER snapshot (GET /api/nodes, cached in store.registry) is
 * AUTHORITATIVE for node-kind knowledge: kinds, ports, props, gates,
 * label_semantic and features. This module's static copy is only the
 * offline bootstrap for the UI (Palette cards / port types / connect
 * validation) when GET /api/nodes fails or has not returned yet —
 * store.registryError marks that degraded state ("stale") in the UI.
 */

/** Registry snapshot port entry from GET /api/nodes. */
export interface PortSpec {
  name: string;
  direction: "in" | "out";
  type: string;
  "burst-format"?: string | null;
}

/**
 * Per-kind node spec from GET /api/nodes. label_semantic / features are
 * the T3 registry-promotion additions (optional + defensively accessed:
 * the frontend must stay green against a snapshot that does not ship
 * them yet — with no features the infer chat button simply does not
 * render until the backend starts sending the field).
 */
export interface NodeSpec {
  ports: PortSpec[];
  props: string[];
  numeric_props: string[];
  gate: string | null;
  /** What the node's user-facing label semantically carries, when it is
   * not a plain display name (dataset: the HF dataset name drives the
   * config mapping). */
  label_semantic?: string | null;
  /** Registry-declared UI feature flags (e.g. "webui-chat-button" on
   * infer) — UI chrome keys off these, never off kind strings. */
  features?: string[];
}

export interface RegistrySnapshot {
  valid_kinds: string[];
  nodes: Record<string, NodeSpec>;
  gates: Record<string, string | null>;
}

/**
 * label_semantic value meaning "this label is the HF dataset name" (the
 * MVP config mapping reads the dataset node's label as the dataset
 * input). Named once here so UI code compares against data, not a
 * kind-string literal.
 */
export const DATASET_LABEL_SEMANTIC = "hf-dataset-name";

/** Feature flag declaring the "Open in webui Chat" render-only shortcut
 * on the node body (currently declared by infer only). */
export const CHAT_BUTTON_FEATURE = "webui-chat-button";

/** Static fallback registry copy — fallback ONLY (see file header). */
export const FALLBACK_REGISTRY: RegistrySnapshot = {
  valid_kinds: ["dataset", "prepare", "tokenize", "train", "eval", "infer"],
  nodes: {
    dataset: {
      ports: [{ name: "cleaned", direction: "out", type: "raw-dir" }],
      props: [],
      numeric_props: [],
      gate: null,
      label_semantic: DATASET_LABEL_SEMANTIC,
      features: [],
    },
    prepare: {
      ports: [
        { name: "raw-dir", direction: "in", type: "raw-dir" },
        { name: "cleaned-dir", direction: "out", type: "cleaned-dir" },
      ],
      props: ["min_chars", "rows", "val_fraction"],
      numeric_props: ["min_chars", "rows", "val_fraction"],
      gate: null,
      label_semantic: null,
      features: [],
    },
    tokenize: {
      ports: [
        { name: "cleaned-dir", direction: "in", type: "cleaned-dir" },
        { name: "shard-dir", direction: "out", type: "shard-dir" },
      ],
      props: ["seq_len", "vocab"],
      numeric_props: ["seq_len", "vocab"],
      gate: null,
      label_semantic: null,
      features: [],
    },
    train: {
      ports: [
        { name: "shard-dir", direction: "in", type: "shard-dir" },
        { name: "ckpt-dir", direction: "out", type: "ckpt-dir" },
      ],
      props: ["lr_preset", "preset", "steps"],
      numeric_props: ["steps"],
      gate: "gpu",
      label_semantic: null,
      features: [],
    },
    eval: {
      ports: [
        { name: "ckpt-dir", direction: "in", type: "ckpt-dir" },
        { name: "report", direction: "out", type: "report" },
      ],
      props: [],
      numeric_props: [],
      gate: "gpu",
      label_semantic: null,
      features: [],
    },
    infer: {
      ports: [{ name: "ckpt-dir", direction: "in", type: "ckpt-dir" }],
      props: [],
      numeric_props: [],
      gate: "gpu/cpu",
      label_semantic: null,
      features: [CHAT_BUTTON_FEATURE],
    },
  },
  gates: { dataset: null, prepare: null, tokenize: null, train: "gpu", eval: "gpu", infer: "gpu/cpu" },
};
