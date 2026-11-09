import type { Edge, Node, NodeChange, EdgeChange, Connection } from "@xyflow/react";
import { applyNodeChanges } from "@xyflow/react";
import { api, errorMessage, type FlowDocument } from "../api";
import { FALLBACK_REGISTRY, type PortSpec, type RegistrySnapshot } from "../nodes/registry";
import { portTypesCompatible } from "../nodes/portTypes";
// Task M2.2 node search: pure matcher used by toXYNodes to stamp the
// node-highlight / node-dim classNames (type-only Graph import is erased
// at runtime - no import cycle with this module).
import { searchMatches } from "./search";
// Toast-truncation helper (lives in the ui slice); type-only FlowState
// import is erased at runtime, so none of this creates a load cycle with
// the composed store shim.
import { truncateErrorList } from "./uiStore";
// FlowState (the composed store state) is type-only imported from ../store;
// erased at runtime, so there is no import cycle.
const DRAFT_KEY = "flow:draft";

import type { FlowState } from "../store";

/**
 * Canonical graph shape is the flow/0.1 document graph (what the backend
 * serves/stores). xyflow ReactFlow Node/Edge objects are a projection of
 * this; mappers live at the bottom of this file.
 *
 * Task 1 store split: this module owns the graph document slice of the old
 * store.ts — types, pure reducers, xyflow mappers, and the persistence
 * actions — composed into the single `useFlowStore` via createGraphSlice.
 */
export interface DomainPosition { x: number; y: number }

export interface DomainNode {
  id: string;
  kind: string;
  label?: string;
  props: Record<string, unknown>;
  position: DomainPosition;
}

export interface DomainEdge {
  id: string;
  from: string;
  to: string;
  fromPort: string;
  toPort: string;
}

export interface Graph {
  nodes: DomainNode[];
  edges: DomainEdge[];
}

/* ------------------------------------------------------------------ */
/* Pure reducers / mappers - exported for direct unit testing.         */
/* ------------------------------------------------------------------ */

/** Deterministic unique id: reuse a simple counter over existing "nK". */
export function nextNodeId(graph: Graph, prefix = "n"): string {
  let max = 0;
  for (const node of graph.nodes) {
    if (!node.id.startsWith(prefix)) continue;
    const rest = node.id.slice(prefix.length);
    if (/^[0-9]+$/.test(rest)) max = Math.max(max, Number(rest));
  }
  return prefix + (max + 1);
}

/** Deterministic unique EDGE id: mirror of nextNodeId over graph.edges.
 * connectReducer used to call nextNodeId(graph, "e"), which only scans
 * graph.nodes, so every new edge got id "e1" (T8 review CRITICAL-1). */
export function nextEdgeId(graph: Graph, prefix = "e"): string {
  let max = 0;
  for (const edge of graph.edges) {
    if (!edge.id.startsWith(prefix)) continue;
    const rest = edge.id.slice(prefix.length);
    if (/^[0-9]+$/.test(rest)) max = Math.max(max, Number(rest));
  }
  return prefix + (max + 1);
}

/** Validate a new edge against the registry snapshot.
 * Returns [] when acceptable, otherwise one human-readable reason. */
export function validateConnect(
  registry: RegistrySnapshot,
  graph: Graph,
  fromId: string,
  toId: string,
  fromPort: string,
  toPort: string,
): string[] {
  const source = graph.nodes.find((n) => n.id === fromId);
  const target = graph.nodes.find((n) => n.id === toId);
  if (!source || !target) return ["edge: both nodes must exist"];
  const srcSpec = registry.nodes[source.kind];
  const dstSpec = registry.nodes[target.kind];
  if (!srcSpec || !dstSpec) return ["edge: unknown node kind"];
  const out = srcSpec.ports.find((p) => p.name === fromPort && p.direction === "out");
  const inP = dstSpec.ports.find((p) => p.name === toPort && p.direction === "in");
  if (!out) return ["edge: '" + fromPort + "' is not an OUT port of kind '" + source.kind + "'"];
  if (!inP) return ["edge: '" + toPort + "' is not an IN port of kind '" + target.kind + "'"];
  if (!portTypesCompatible(out.type, inP.type)) {
    return [
      "edge: port types do not match ('" + out.type + "' on '" + source.kind + "." + fromPort +
      "' vs '" + inP.type + "' on '" + target.kind + "." + toPort + "')",
    ];
  }
  return [];
}

/** Invalid connects must leave the graph untouched (passes through). */
export function connectReducer(
  registry: RegistrySnapshot,
  graph: Graph,
  c: Connection,
): { graph: Graph; error: string | null } {
  const fromPort = c.sourceHandle;
  const toPort = c.targetHandle;
  if (!c.source || !c.target || !fromPort || !toPort) {
    return { graph, error: "edge: connection needs both handles" };
  }
  const reasons = validateConnect(registry, graph, c.source, c.target, fromPort, toPort);
  if (reasons.length > 0) return { graph, error: reasons[0] };
  const edge: DomainEdge = {
    id: nextEdgeId(graph),
    from: c.source,
    to: c.target,
    fromPort,
    toPort,
  };
  return { graph: { ...graph, edges: [...graph.edges, edge] }, error: null };
}

/** Default props for a freshly added node: the server applies its own
 * defaults (per /api/nodes registry), so an empty {} object - which every
 * kind accepts by /api/validate - is the registry-consistent default. */
export function addNodeReducer(
  registry: RegistrySnapshot,
  graph: Graph,
  kind: string,
  position: DomainPosition,
): { graph: Graph; id: string | null; error: string | null } {
  if (!registry.nodes[kind]) {
    return { graph, id: null, error: "unknown node kind '" + kind + "'" };
  }
  const node: DomainNode = {
    id: nextNodeId(graph, "n"),
    kind,
    props: {},
    position: { x: position.x, y: position.y },
  };
  return { graph: { ...graph, nodes: [...graph.nodes, node] }, id: node.id, error: null };
}

/* ------------------------------------------------------------------ */
/* Task 9: props editing + flow name slug helpers                     */
/* ------------------------------------------------------------------ */

/** Build the flow/0.1 PUT document for the current graph. The document
 * meta name echoes the slug under which the graph is saved server-side. */
export function flowDocument(name: string, graph: Graph): FlowDocument {
  return { schema: "flow/0.1", meta: { name }, graph };
}

/** Client-side slug validation for the Save/Open name inputs
 * ([a-z0-9-]{1,64}). The server re-validates authoritatively; this only
 * gives immediate feedback and avoids a doomed request round-trip. */
export const FLOW_SLUG_RE = /^[a-z0-9-]{1,64}$/;

export function isValidFlowName(name: string): boolean {
  return FLOW_SLUG_RE.test(name);
}

/** Pure props write-back: replace node.props in place-by-id and return a
 * NEW graph (validator-agnostic here — the server validates on save and
 * on POST /api/validate). Unknown ids return the original graph object. */
export function updateNodePropsReducer(
  graph: Graph,
  id: string,
  props: Record<string, unknown>,
): Graph {
  if (!graph.nodes.some((n) => n.id === id)) return graph;
  return {
    ...graph,
    nodes: graph.nodes.map((n) => (n.id === id ? { ...n, props } : n)),
  };
}

/** Pure label write-back (browser drill finding): the MVP config mapping
 * reads the dataset node's label as the HF dataset name, so the label
 * must be editable even when the registry lists no prop widgets for the
 * kind. Unknown ids return the original graph object. */
export function updateNodeLabelReducer(
  graph: Graph,
  id: string,
  label: string,
): Graph {
  if (!graph.nodes.some((n) => n.id === id)) return graph;
  const trimmed = label.trim();
  return {
    ...graph,
    nodes: graph.nodes.map((n) => {
      if (n.id !== id) return n;
      const next = { ...n };
      if (trimmed === "") delete next.label;
      else next.label = trimmed;
      return next;
    }),
  };
}

/* ------------------------------------------------------------------ */
/* xyflow <-> domain mapping                                          */
/* ------------------------------------------------------------------ */

/** Shared empty features list for snapshots/kinds without the field
 * (identity-stable placeholder for the data-identity check). */
const EMPTY_FEATURES: string[] = [];

export interface PipelineNodeData extends Record<string, unknown> {
  kind: string;
  label?: string;
  props: Record<string, unknown>;
  ports: PortSpec[];
  /** Registry feature flags for this node's kind (T3: the infer node's
   * chat button keys off this data, never off kind strings). */
  features: string[];
  /** Registry label semantics for the kind (dataset: "hf-dataset-name"). */
  labelSemantic: string | null;
  runState?: "idle" | "running" | "done" | "error";
  /** Backend exit_code, only shown/tooltiped in terminal dot states. */
  exitCode?: number | null;
}

export type PipelineNode = Node<PipelineNodeData, "pipeline">;

/**
 * runState/exitCode threading (Task 10, honest MVP simplification): the
 * backend runs one whole-graph pipeline — nodes are NOT individually
 * resolved. So while runStatus.state === "running" EVERY node's dot
 * shows "running"; at a terminal done/error the dots flip set-wide and
 * the title tooltip carries the backend exit_code. Per-node phases are
 * a future engine feature; until then this is deliberately global, not
 * faked per node.
 */
export function toXYNodes(
  graph: Graph,
  registry: RegistrySnapshot,
  prev?: PipelineNode[],
  runState: "idle" | "running" | "done" | "error" = "idle",
  exitCode: number | null = null,
  /** Task M2.2 node search: when non-empty, matching nodes get the
   * "node-highlight" class and all others "node-dim" (both consumed by
   * App.css). Empty/whitespace = no search active, classNames omitted. */
  searchQuery?: string,
): PipelineNode[] {
  const prevById = new Map((prev ?? []).map((n) => [n.id, n]));
  const matchIds = new Set(searchMatches(graph, searchQuery ?? ""));
  return graph.nodes.map((n) => {
    // Data identity: when the domain node is unchanged (same kind/label
    // props/ports object references) reuse the previous data object so the
    // memoized PipelineNode does not re-render on unrelated updates.
    const p = prevById.get(n.id);
    const spec = registry.nodes[n.kind];
    const ports = spec?.ports ?? [];
    // T3 registry promotion: per-kind UI data (feature flags, label
    // semantics) is threaded from the snapshot into node data, so the
    // renderers stay free of kind-string conditionals.
    const features = spec?.features ?? EMPTY_FEATURES;
    const labelSemantic = spec?.label_semantic ?? null;
    const stableData =
      p !== undefined &&
      p.data.kind === n.kind &&
      p.data.label === n.label &&
      p.data.props === n.props &&
      p.data.ports === ports &&
      p.data.features === features &&
      p.data.labelSemantic === labelSemantic &&
      p.data.runState === runState &&
      (p.data.exitCode ?? null) === exitCode;
    const data: PipelineNodeData = stableData
      ? p.data
      : {
          kind: n.kind,
          ...(n.label !== undefined ? { label: n.label } : {}),
          props: n.props,
          ports,
          features,
          labelSemantic,
          runState,
          exitCode,
        };
    return {
      id: n.id,
      type: "pipeline" as const,
      position: n.position,
      // Keep measured geometry + interaction flags; the domain node (a
      // flow/0.1 document shape) must not absorb xyflow-only fields.
      // measured is REQUIRED here, not optional polish: React Flow keeps
      // the node wrapper at visibility:hidden until the user node itself
      // carries measured — dropping it made every node invisible and
      // unhittable from the second render onward (browser drill finding).
      measured: p?.measured,
      width: p?.width,
      height: p?.height,
      dragging: p?.dragging,
      selected: p?.selected,
      // Task M2.2: search highlight classes land on the node WRAPPER
      // (React Flow applies node.className there), never in node data -
      // the data identity above must stay search-agnostic.
      className: matchIds.size > 0 ? (matchIds.has(n.id) ? "node-highlight" : "node-dim") : undefined,
      data,
    };
  });
}

export function toXYEdges(graph: Graph): Edge[] {
  return graph.edges.map((e) => ({
    id: e.id,
    source: e.from,
    target: e.to,
    sourceHandle: e.fromPort,
    targetHandle: e.toPort,
  }));
}

/** Apply an already-projected set of xyflow nodes back onto the domain
 * graph: positions updated, deleted nodes removed, kind/props/label
 * preserved from the existing domain node. xyflow-only fields
 * (width/height/dragging/selected) intentionally never land on the
 * domain node — they persist on the store projection (see toXYNodes). */
export function fromXYNodes(nodes: Node[], existing: Graph): Graph {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const keep = existing.nodes.filter((n) => byId.has(n.id));
  const changed = keep.map((n) => {
    const xy = byId.get(n.id)!;
    if (xy.position.x === n.position.x && xy.position.y === n.position.y) return n;
    return { ...n, position: { x: xy.position.x, y: xy.position.y } };
  });
  return { nodes: changed, edges: existing.edges };
}

/* ------------------------------------------------------------------ */
/* Graph document slice state                                          */
/* ------------------------------------------------------------------ */

/** Slice-creator signatures: the composed store's set/get, as zustand
 * passes them to creator functions. */
import type { StoreApi } from "zustand";
type SetFlow = StoreApi<FlowState>["setState"];
type GetFlow = StoreApi<FlowState>["getState"];

export interface GraphState {
  graph: Graph;
  /** Last xyflow projection (full nodes, incl. measured width/height and
   * dragging/selected flags). toXYNodes merges it back so unchanged
   * domain nodes keep their data identity (PipelineNode is memoized) and
   * geometry/interaction flags persist across the controlled-render
   * round-trip (T8 review IMPORTANT-3). */
  projection: PipelineNode[];
  /** Slug of the graph as saved/opened (null while purely unsaved). Run
   * wiring PUT-saves under this name; Validate uses it for meta.name
   * (choice: an unsaved graph validates as meta.name "untitled" —
   * TOCTOU-free, the server validates the document, not the slug). */
  currentFlowName: string | null;
  /** Graph document the server last accepted via POST /api/validate —
   * the raw material for the read-only Preview summary card. */
  validatedDoc: FlowDocument | null;
  setGraph: (g: Graph) => void;
  /** Task 9 inspector props write-back (see updateNodePropsReducer). */
  updateNodeProps: (id: string, props: Record<string, unknown>) => void;
  /** Label write-back (see updateNodeLabelReducer) — dataset nodes need it
   * for the MVP config mapping (label = HF dataset name). */
  updateNodeLabel: (id: string, label: string) => void;
  /** Task 9 file actions: Open = GET /api/flows/{name} -> setGraph (which
   * also resets the xyflow projection via setGraph); Save = PUT
   * /api/flows/{name}. Both surface errors through the existing
   * store.error toast channel and never leave a half-applied graph.
   * saveFlow keeps the collapsed names awaitable for the toolbar modal. */
  openFlow: (name: string) => Promise<boolean>;
  saveFlow: (name: string) => Promise<boolean>;
  /** Task M2.4: delete the saved flow server-side (DELETE
   * /api/flows/{name}), then reset the store to a fresh unsaved graph —
   * the deleted flow is no longer the source of truth. Success toasts
   * via the "OK:" convention; failures land in the toast channel and
   * leave the client graph untouched. */
  deleteFlow: (name: string) => Promise<boolean>;
  /** Task 10 wiring: validate the UNSAVED graph (POST /api/validate). */
  validateFlow: () => Promise<boolean>;
  /** Pure, reducer-safe actions (each fully testable via getState()). */
  addNode: (kind: string, position: DomainPosition) => string | null;
  applyNodesChanges: (changes: NodeChange[]) => void;
  applyEdgesChanges: (changes: EdgeChange[]) => void;
  connect: (c: Connection) => boolean;
}

/** Graph document slice of the composed flow store. Anything not in
 * GraphState above (registry/selection/run) is read through the composed
 * get()/set() so the slice stays composable and behavior-identical. */
export function createGraphSlice(set: SetFlow, get: GetFlow): GraphState {
  return {
    graph: { nodes: [], edges: [] },
    projection: [],
    currentFlowName: null,
    validatedDoc: null,

    setGraph: (graph) => {
      const g = get();
      get().push(g.graph, "replace graph");
      set({ graph, projection: [], validatedDoc: null });
    },
    updateNodeLabel: (id, label) => {
      const g = get();
      const next = updateNodeLabelReducer(g.graph, id, label);
      if (next !== g.graph) {
        get().push(g.graph, "edit label");
        set({ graph: next, validatedDoc: null });
      }
    },
    updateNodeProps: (id, props) => {
      const g = get();
      const next = updateNodePropsReducer(g.graph, id, props);
      // Unknown id keeps the same graph object -> no state churn.
      // Any real graph edit invalidates the cached validatedDoc so the
      // Preview card can never claim freshness over a stale document
      // (T10 review IMPORTANT-2).
      if (next !== g.graph) {
        get().push(g.graph, "edit props");
        set({ graph: next, validatedDoc: null });
      }
    },
    openFlow: async (name) => {
      try {
        const doc = await api.getFlow(name);
        // Clear the draft: the server is now the source of truth (M1.2)
        try { localStorage.removeItem(DRAFT_KEY); } catch { /* ignore */ }
        set({ graph: doc.graph, projection: [], error: null, currentFlowName: name, validatedDoc: null, past: [], future: [] });
        return true;
      } catch (e) {
        set({ error: errorMessage(e) });
        return false;
      }
    },
    saveFlow: async (name) => {
      if (!isValidFlowName(name)) {
        set({
          error: "flow name must match [a-z0-9-]{1,64} (lowercase letters, digits, dashes)",
        });
        return false;
      }
      try {
        await api.saveFlow(name, flowDocument(name, get().graph));
        // Clear the draft: the server is now the source of truth (M1.2)
        try { localStorage.removeItem(DRAFT_KEY); } catch { /* ignore */ }
        set({ error: null, currentFlowName: name });
        return true;
      } catch (e) {
        set({ error: errorMessage(e) });
        return false;
      }
    },
    deleteFlow: async (name) => {
      try {
        await api.deleteFlow(name);
        // The deleted flow is no longer the source of truth: fresh
        // unsaved graph, no saved slug, no history (mirrors openFlow's
        // reset — a deleted flow is not undoable back into existence).
        set({
          graph: { nodes: [], edges: [] },
          projection: [],
          validatedDoc: null,
          currentFlowName: null,
          past: [],
          future: [],
          error: "OK: deleted flow '" + name + "'",
        });
        return true;
      } catch (e) {
        set({ error: errorMessage(e) });
        return false;
      }
    },

    // Task 10 run wiring -------------------------------------------------
    // Validate ALWAYS posts the unsaved graph: the doc meta.name is the
    // saved slug when there is one, else the literal "untitled" (documented
    // choice above). A successful response caches validatedDoc for the
    // Preview summary; errors go to the toast, first 3 + "+N more".
    validateFlow: async () => {
      const name = get().currentFlowName;
      const doc = flowDocument(
        name !== null && isValidFlowName(name) ? name : "untitled",
        get().graph,
      );
      try {
        const r = await api.validateFlow(doc);
        if (r.ok) {
          // Distinct success marker on the toast channel ("OK:" prefix) so
          // a passing validate is VISIBLE, not a silent null (T10 review
          // MISSING-1). The prefix distinguishes it from error toasts.
          set({ validatedDoc: doc, error: "OK: validation passed — document accepted" });
          return true;
        }
        set({ validatedDoc: null, error: truncateErrorList(r.errors).join(" · ") });
        return false;
      } catch (e) {
        set({ validatedDoc: null, error: errorMessage(e) });
        return false;
      }
    },

    addNode: (kind, position) => {
      const g = get();
      const registry = g.registry ?? FALLBACK_REGISTRY;
      const res = addNodeReducer(registry, g.graph, kind, position);
      if (res.error) {
        set({ error: res.error });
        return null;
      }
      get().push(g.graph, "add " + kind);
      set({ graph: res.graph, validatedDoc: null });
      return res.id;
    },

    applyNodesChanges: (changes) => {
      const g = get();
      // Only push history for remove changes (not position/selection drags)
      const hasRemove = changes.some((c) => c.type === "remove");
      if (hasRemove) get().push(g.graph, "delete node");
      const xyNodes = applyNodeChanges(
        changes,
        toXYNodes(g.graph, g.registry ?? FALLBACK_REGISTRY, g.projection),
      ) as PipelineNode[];
      const graph = fromXYNodes(xyNodes, g.graph);
      // Mirror select changes into store.selection: the last select change
      // wins (selected:false clears — e.g. the pane click deselection).
      let selection: string | null | undefined;
      for (const c of changes) {
        if (c.type === "select") selection = c.selected ? c.id : null;
      }
      // Carry-over T9 finding (b): a remove change that deletes the node the
      // Inspector is currently editing must clear the stale selection, or the
      // Properties panel would keep rendering a ghost node's props.
      if (selection === undefined && get().selection !== null) {
        const alive = new Set(graph.nodes.map((n) => n.id));
        if (!alive.has(get().selection as string)) selection = null;
      }
      set({
        graph,
        projection: xyNodes,
        validatedDoc: null,
        ...(selection === undefined ? {} : { selection }),
      });
    },

    applyEdgesChanges: (changes) => {
      // Only "remove" changes can alter the domain graph; every other
      // xyflow edge change kind (select/dimensions/...) is projection-only.
      // Removals are counted per id (NOT a keep-Set, which would collapse
      // duplicate-id edges into one removal) so exactly one domain edge per
      // requested remove disappears (T8 review IMPORTANT-2).
      const removals = new Map<string, number>();
      for (const c of changes) {
        if (c.type === "remove") removals.set(c.id, (removals.get(c.id) ?? 0) + 1);
      }
      if (removals.size === 0) return;
      const g = get();
      get().push(g.graph, "delete edge");
      const graph = g.graph;
      const edges = graph.edges.filter((e) => {
        const pending = removals.get(e.id) ?? 0;
        if (pending <= 0) return true;
        removals.set(e.id, pending - 1);
        return false;
      });
      set({ graph: { ...graph, edges }, validatedDoc: null });
    },

    connect: (c: Connection) => {
      const g = get();
      const registry = g.registry ?? FALLBACK_REGISTRY;
      const res = connectReducer(registry, g.graph, c);
      if (res.error) {
        set({ error: res.error });
        return false;
      }
      get().push(g.graph, "connect");
      set({ graph: res.graph, validatedDoc: null });
      return true;
    },
  };
}