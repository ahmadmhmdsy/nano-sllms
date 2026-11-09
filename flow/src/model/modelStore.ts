/**
 * Model-tab store (F2 plan Task 5).
 *
 * A dedicated zustand store for the Model mode — deliberately separate
 * from useFlowStore (the pipeline store): the two modes own independent
 * graphs, selections and file names, and sharing one store would thread
 * mode conditionals through every pipeline action.
 *
 * Same layering as stores/graphStore: domain graph (model/0.1 document
 * shape), pure reducers exported for unit testing, xyflow mappers, and
 * store actions. The spec-driven walk cache (shapes/params/total/errors)
 * recomputes on EVERY graph change — the graph is small, and stale
 * readouts are the one dishonesty the Model tab must never show.
 *
 * NO kind-string conditionals anywhere: every kind-specific behavior
 * (ports, props, defaults, labels) comes from LAYER_REGISTRY specs.
 */
import { create } from "zustand";
import type { Edge, Node, NodeChange, EdgeChange, Connection } from "@xyflow/react";
import { applyNodeChanges } from "@xyflow/react";
import type { ModelDocument } from "../api";
import { api, errorMessage } from "../api";
import type { DomainEdge, DomainNode, DomainPosition } from "../stores/graphStore";
import { MODEL_KINDS, LAYER_REGISTRY, type LayerSpec, type Shape } from "./registry";
import { walkModelGraph, type WalkResult } from "./graphWalk";

/* ------------------------------------------------------------------ */
/* Pure helpers (exported for unit testing)                            */
/* ------------------------------------------------------------------ */

/** Deterministic unique node id over existing "nK" ids (same scheme as
 * the pipeline graphStore's nextNodeId — max counter + 1). */
export function nextNodeId(nodes: DomainNode[], prefix = "n"): string {
  let max = 0;
  for (const node of nodes) {
    if (!node.id.startsWith(prefix)) continue;
    const rest = node.id.slice(prefix.length);
    if (/^[0-9]+$/.test(rest)) max = Math.max(max, Number(rest));
  }
  return prefix + (max + 1);
}

/** Deterministic unique EDGE id over existing "eK" ids. */
export function nextEdgeId(edges: DomainEdge[], prefix = "e"): string {
  let max = 0;
  for (const edge of edges) {
    if (!edge.id.startsWith(prefix)) continue;
    const rest = edge.id.slice(prefix.length);
    if (/^[0-9]+$/.test(rest)) max = Math.max(max, Number(rest));
  }
  return prefix + (max + 1);
}

/** Default props for a freshly added node: every spec prop with a
 * declared default is seeded (registry-driven; props without defaults
 * stay absent so inferShape reports the honest "must be a finite
 * number" error instead of inventing a value). */
export function defaultProps(spec: LayerSpec): Record<string, unknown> {
  const props: Record<string, unknown> = {};
  for (const p of spec.props) {
    if (p.default !== undefined) props[p.name] = p.default;
  }
  return props;
}

/** All reasons a candidate edge is rejected, else []. Spec-driven only:
 * 1. ports must exist on the source/target kinds (right direction);
 * 2. a target input port accepts EXACTLY ONE edge — a second edge into
 *    the same (node, port) pair is refused (output ports may fan out);
 * 3. self-loops are refused (the walker reports them as cycles).
 */
export function validateConnect(
  nodes: DomainNode[],
  edges: DomainEdge[],
  fromId: string,
  fromPort: string,
  toId: string,
  toPort: string,
): string[] {
  if (fromId === toId) return ["edge: self-loops are not allowed"];
  const source = nodes.find((n) => n.id === fromId);
  const target = nodes.find((n) => n.id === toId);
  if (!source || !target) return ["edge: both nodes must exist"];
  const srcSpec = LAYER_REGISTRY[source.kind as keyof typeof LAYER_REGISTRY];
  const dstSpec = LAYER_REGISTRY[target.kind as keyof typeof LAYER_REGISTRY];
  if (!srcSpec || !dstSpec) return ["edge: unknown node kind"];
  if (!srcSpec.outputs.some((p) => p.name === fromPort)) {
    return ["edge: '" + fromPort + "' is not an output port of kind '" + source.kind + "'"];
  }
  if (!dstSpec.inputs.some((p) => p.name === toPort)) {
    return ["edge: '" + toPort + "' is not an input port of kind '" + target.kind + "'"];
  }
  if (edges.some((e) => e.to === toId && e.toPort === toPort)) {
    return [
      "edge: input port '" + toPort + "' of node '" + toId +
      "' already has a connection (each input accepts exactly one edge)",
    ];
  }
  return [];
}

/** Connection reducer: validates then appends; a rejected edge leaves the
 * graph untouched (error message returned for the toast). */
export function connectReducer(
  nodes: DomainNode[],
  edges: DomainEdge[],
  c: Connection,
): { edges: DomainEdge[]; error: string | null } {
  const fromPort = c.sourceHandle;
  const toPort = c.targetHandle;
  if (!c.source || !c.target || !fromPort || !toPort) {
    return { edges, error: "edge: connection needs both handles" };
  }
  const reasons = validateConnect(nodes, edges, c.source, fromPort, c.target, toPort);
  if (reasons.length > 0) return { edges, error: reasons[0] };
  const edge: DomainEdge = {
    id: nextEdgeId(edges),
    from: c.source,
    to: c.target,
    fromPort,
    toPort,
  };
  return { edges: [...edges, edge], error: null };
}

/** Pure props write-back by id; unknown ids keep the original array. */
export function updatePropsReducer(
  nodes: DomainNode[],
  id: string,
  props: Record<string, unknown>,
): DomainNode[] {
  if (!nodes.some((n) => n.id === id)) return nodes;
  return nodes.map((n) => (n.id === id ? { ...n, props } : n));
}

/** Apply already-projected xyflow nodes back onto the domain graph:
 * positions updated, deleted nodes removed. Node deletion also drops the
 * node's edges (a dangling edge would just be walk noise otherwise). */
export function fromXYNodes(
  xyNodes: { id: string; position: { x: number; y: number } }[],
  existing: DomainNode[],
  existingEdges: DomainEdge[],
): { nodes: DomainNode[]; edges: DomainEdge[] } {
  const byId = new Map(xyNodes.map((n) => [n.id, n]));
  const nodes = existing
    .filter((n) => byId.has(n.id))
    .map((n) => {
      const xy = byId.get(n.id)!;
      if (xy.position.x === n.position.x && xy.position.y === n.position.y) return n;
      return { ...n, position: { x: xy.position.x, y: xy.position.y } };
    });
  const removed = new Set(existing.filter((n) => !byId.has(n.id)).map((n) => n.id));
  const edges = existingEdges.filter((e) => !removed.has(e.from) && !removed.has(e.to));
  return { nodes, edges };
}

/** Model graph = the model/0.1 body (DomainNode/DomainEdge reused from
 * the pipeline graphStore — same field shape, one declaration). */
export interface ModelGraph {
  nodes: DomainNode[];
  edges: DomainEdge[];
}

/** Serialize to the model/0.1 document (server Task 4 contract:
 * format/name/nodes[{id,kind,props,position}]/edges[{id,from,to,fromPort,toPort}]/meta). */
export function modelDocument(name: string, graph: ModelGraph): ModelDocument {
  return {
    format: "model/0.1",
    name,
    nodes: graph.nodes.map((n) => ({
      id: n.id,
      kind: n.kind,
      props: n.props,
      position: n.position,
    })),
    edges: graph.edges.map((e) => ({
      id: e.id,
      from: e.from,
      to: e.to,
      fromPort: e.fromPort,
      toPort: e.toPort,
    })),
    meta: {},
  };
}

/* ------------------------------------------------------------------ */
/* xyflow mappers                                                      */
/* ------------------------------------------------------------------ */

export interface ModelNodeData extends Record<string, unknown> {
  kind: string;
  label: string;
  props: Record<string, unknown>;
  inputs: { name: string; label: string }[];
  outputs: { name: string; label: string }[];
  /** Walk cache readout for this node: null when errored/missing. */
  shape: Shape | null;
  params: number | null;
  /** First walk error for the node (drives the red badge + tooltip). */
  error: string | null;
}

export type ModelNode = Node<ModelNodeData, "model">;

/** Project domain nodes into xyflow nodes, threading the walk cache so
 * ModelNode stays render-only. Spec-driven: label and port lists come
 * from LAYER_REGISTRY — no kind conditionals. */
export function toXYNodes(graph: ModelGraph, walk: WalkResult): ModelNode[] {
  return graph.nodes.map((n) => {
    const spec = LAYER_REGISTRY[n.kind as keyof typeof LAYER_REGISTRY];
    const nodeErrors = walk.errors.filter((e) => e.nodeId === n.id);
    return {
      id: n.id,
      type: "model" as const,
      position: n.position,
      data: {
        kind: n.kind,
        label: spec?.label ?? n.kind,
        props: n.props,
        inputs: spec?.inputs ?? [],
        outputs: spec?.outputs ?? [],
        shape: walk.shapes.get(n.id) ?? null,
        params: walk.params[n.id] ?? null,
        error: nodeErrors.length > 0 ? nodeErrors.map((e) => e.message).join("; ") : null,
      },
    };
  });
}

export function toXYEdges(edges: DomainEdge[]): Edge[] {
  return edges.map((e) => ({
    id: e.id,
    source: e.from,
    target: e.to,
    sourceHandle: e.fromPort,
    targetHandle: e.toPort,
  }));
}

/* ------------------------------------------------------------------ */
/* Composed store                                                      */
/* ------------------------------------------------------------------ */

export interface ModelState extends ModelGraph {
  /** Model name (the slug used for GET/POST /api/models/{name}). */
  name: string;
  /** True when the current graph has edits since the last save/open. */
  dirty: boolean;
  /** Cached spec walk; recomputed on every graph change. */
  walk: WalkResult;
  /** xyflow projection (positions/selection live here, never on domain). */
  projection: ModelNode[];
  /** Id of the node the ModelInspector is editing. */
  selection: string | null;
  /** Toast channel (same behavior as the pipeline store's error). */
  error: string | null;

  addNode: (kind: string, position?: DomainPosition) => string | null;
  deleteNode: (id: string) => void;
  applyNodesChanges: (changes: NodeChange[]) => void;
  applyEdgesChanges: (changes: EdgeChange[]) => void;
  connect: (c: Connection) => boolean;
  updateProps: (id: string, props: Record<string, unknown>) => void;
  openDoc: (doc: ModelDocument) => void;
  setModelName: (name: string) => void;
  setSelection: (id: string | null) => void;
  setError: (e: string | null) => void;
  /** POST /api/models/{name}; resolves false on failure (toast set). */
  save: (name: string) => Promise<boolean>;
  /** GET /api/models/{name} then openDoc; resolves false on failure. */
  open: (name: string) => Promise<boolean>;
}

/** Compute the walk over the given graph (single call site for the cache). */
function computeWalk(graph: ModelGraph): WalkResult {
  return walkModelGraph(graph, LAYER_REGISTRY);
}

export const useModelStore = create<ModelState>()((set, get) => ({
  nodes: [],
  edges: [],
  name: "",
  dirty: false,
  walk: { shapes: new Map(), params: {}, total: 0, errors: [] },
  projection: [],
  selection: null,
  error: null,

  addNode: (kind, position) => {
    const spec = LAYER_REGISTRY[kind as keyof typeof LAYER_REGISTRY];
    if (!spec) {
      set({ error: "unknown layer kind '" + kind + "'" });
      return null;
    }
    const nodes = get().nodes;
    const id = nextNodeId(nodes);
    const node: DomainNode = {
      id,
      kind,
      props: defaultProps(spec),
      position: position ?? { x: 60 + nodes.length * 24, y: 60 + nodes.length * 24 },
    };
    const nextNodes = [...nodes, node];
    set({
      nodes: nextNodes,
      dirty: true,
      walk: computeWalk({ nodes: nextNodes, edges: get().edges }),
    });
    return id;
  },

  deleteNode: (id) => {
    const { nodes, edges } = get();
    const nextNodes = nodes.filter((n) => n.id !== id);
    const nextEdges = edges.filter((e) => e.from !== id && e.to !== id);
    const selection = get().selection === id ? null : get().selection;
    set({
      nodes: nextNodes,
      edges: nextEdges,
      selection,
      dirty: true,
      walk: computeWalk({ nodes: nextNodes, edges: nextEdges }),
    });
  },

  applyNodesChanges: (changes) => {
    const g = get();
    const xyNodes = applyNodeChanges(changes, toXYNodes(g, g.walk)) as unknown as ModelNode[];
    const applied = fromXYNodes(xyNodes, g.nodes, g.edges);
    // Mirror select changes into store.selection (last select wins).
    let selection: string | null | undefined;
    for (const c of changes) {
      if (c.type === "select") selection = c.selected ? c.id : null;
    }
    /* Task M6.1: the walk is memoized on graph object identity — the
     * store-level equivalent of useMemo(() => walkModelGraph(graph), [graph]).
     * A select-only change round-trips a NEW graph object whose node/edge
     * entries are all identical references (fromXYNodes maps unchanged
     * positions back to the original domain nodes), so the cached walk is
     * reused instead of recomputing the whole topological walk on every
     * selection click. Any real graph mutation changes node/edge identity
     * and recomputes. */
    const graphChanged =
      applied.nodes.length !== g.nodes.length ||
      applied.edges.length !== g.edges.length ||
      applied.nodes.some((n, i) => n !== g.nodes[i]) ||
      applied.edges.some((e, i) => e !== g.edges[i]);
    set({
      nodes: applied.nodes,
      edges: applied.edges,
      projection: xyNodes,
      dirty: true,
      walk: graphChanged ? computeWalk(applied) : g.walk,
      ...(selection === undefined ? {} : { selection }),
    });
  },

  applyEdgesChanges: (changes) => {
    const removals = new Map<string, number>();
    for (const c of changes) {
      if (c.type === "remove") removals.set(c.id, (removals.get(c.id) ?? 0) + 1);
    }
    if (removals.size === 0) return;
    const graph = get();
    const edges = graph.edges.filter((e) => {
      const pending = removals.get(e.id) ?? 0;
      if (pending <= 0) return true;
      removals.set(e.id, pending - 1);
      return false;
    });
    set({
      edges,
      dirty: true,
      walk: computeWalk({ nodes: graph.nodes, edges }),
    });
  },

  connect: (c) => {
    const g = get();
    const res = connectReducer(g.nodes, g.edges, c);
    if (res.error !== null) {
      set({ error: res.error });
      return false;
    }
    set({ edges: res.edges, dirty: true, walk: computeWalk({ nodes: g.nodes, edges: res.edges }) });
    return true;
  },

  updateProps: (id, props) => {
    const g = get();
    const nodes = updatePropsReducer(g.nodes, id, props);
    if (nodes === g.nodes) return; // unknown id: no churn
    set({ nodes, dirty: true, walk: computeWalk({ nodes, edges: g.edges }) });
  },

  openDoc: (doc) => {
    const nodes = doc.nodes.map((n) => ({ ...n }));
    const edges = doc.edges.map((e) => ({ ...e }));
    set({
      nodes,
      edges,
      name: doc.name,
      dirty: false,
      selection: null,
      projection: [],
      walk: computeWalk({ nodes, edges }),
      error: null,
    });
  },

  setModelName: (name) => set({ name, dirty: true }),

  setSelection: (selection) => set({ selection }),
  setError: (error) => set({ error }),

  save: async (name) => {
    try {
      await api.saveModel(name, modelDocument(name, get()));
      set({ name, dirty: false, error: null });
      return true;
    } catch (e) {
      set({ error: errorMessage(e) });
      return false;
    }
  },

  open: async (name) => {
    try {
      const doc = await api.getModel(name);
      get().openDoc(doc);
      return true;
    } catch (e) {
      set({ error: errorMessage(e) });
      return false;
    }
  },
}));

/* Palette order: the registry's own kind list, exported for ModelPalette. */
export const MODEL_PALETTE_KINDS: readonly string[] = MODEL_KINDS;
