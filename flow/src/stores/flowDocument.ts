/* Task M2.3: pure flow/0.1 document serialization + client-side
 * import validation. No DOM, no store: every function here is a plain
 * unit-testable pure function. The Toolbar's Blob download and
 * <input type="file"> picker are the only browser-only pieces and stay
 * thin in App.tsx. */
import type { FlowDocument } from "../api";
import { flowDocument, type DomainEdge, type DomainNode, type Graph } from "./graphStore";

/** Serialize the current graph into a pretty-printed flow/0.1 document
 * string - the exact shape the Export button downloads and the Import
 * picker accepts. Reuses the store's flowDocument builder so the on-disk
 * document is byte-identical to what Save/PUT would send. */
export function serializeFlow(name: string, graph: Graph): string {
  return JSON.stringify(flowDocument(name, graph), null, 2);
}

/** Result of parsing/validating a flow document: exactly one of doc /
 * error is set. */
export interface FlowParseResult {
  doc?: FlowDocument;
  error?: string;
}

/** Parse a .flow.json text and schema-validate it as a flow/0.1
 * document. Pure: malformed JSON, a wrong schema tag, or missing /
 * mistyped fields yield { error } and never a doc. */
export function parseFlowDocument(text: string): FlowParseResult {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { error: "invalid JSON: " + (e instanceof Error ? e.message : String(e)) };
  }
  return validateFlowDocument(raw);
}

function isObject(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function isNonEmptyString(v: unknown): v is string {
  return typeof v === "string" && v.length > 0;
}

/** Schema-check an already-parsed JSON value as a flow/0.1 document and
 * normalize it: the returned doc's graph carries exactly the domain
 * fields (node id/kind/label?/props/position, edge
 * id/from/to/fromPort/toPort), so stray fields from a hand-edited file
 * can never leak into the store projection. */
export function validateFlowDocument(raw: unknown): FlowParseResult {
  if (!isObject(raw)) return { error: "flow document must be a JSON object" };
  if (raw.schema !== "flow/0.1") {
    return {
      error: "unsupported flow schema " + JSON.stringify(raw.schema) + ' (expected "flow/0.1")',
    };
  }
  if (!isObject(raw.meta)) return { error: "flow document is missing the meta object" };
  if (!isNonEmptyString(raw.meta.name)) {
    return { error: "flow document meta.name must be a non-empty string" };
  }
  if (!isObject(raw.graph)) return { error: "flow document is missing the graph object" };
  const { nodes, edges } = raw.graph;
  if (!Array.isArray(nodes)) return { error: "flow document graph.nodes must be an array" };
  if (!Array.isArray(edges)) return { error: "flow document graph.edges must be an array" };

  const outNodes: DomainNode[] = [];
  for (const n of nodes) {
    if (!isObject(n)) return { error: "graph node must be an object" };
    if (!isNonEmptyString(n.id)) return { error: "graph node id must be a non-empty string" };
    if (!isNonEmptyString(n.kind)) return { error: "graph node kind must be a non-empty string" };
    if (!isObject(n.props)) return { error: "graph node props must be an object" };
    if (!isObject(n.position)) return { error: "graph node position must be an object" };
    if (typeof n.position.x !== "number" || typeof n.position.y !== "number") {
      return { error: "graph node position must have numeric x and y" };
    }
    if (n.label !== undefined && typeof n.label !== "string") {
      return { error: "graph node label must be a string" };
    }
    outNodes.push({
      id: n.id,
      kind: n.kind,
      ...(n.label !== undefined ? { label: n.label } : {}),
      props: n.props,
      position: { x: n.position.x, y: n.position.y },
    });
  }

  const outEdges: DomainEdge[] = [];
  for (const e of edges) {
    if (!isObject(e)) return { error: "graph edge must be an object" };
    if (!isNonEmptyString(e.id)) return { error: "graph edge id must be a non-empty string" };
    if (!isNonEmptyString(e.from)) return { error: "graph edge from must be a non-empty string" };
    if (!isNonEmptyString(e.to)) return { error: "graph edge to must be a non-empty string" };
    if (!isNonEmptyString(e.fromPort)) return { error: "graph edge fromPort must be a non-empty string" };
    if (!isNonEmptyString(e.toPort)) return { error: "graph edge toPort must be a non-empty string" };
    outEdges.push({ id: e.id, from: e.from, to: e.to, fromPort: e.fromPort, toPort: e.toPort });
  }

  return {
    doc: {
      schema: "flow/0.1",
      meta: { name: raw.meta.name },
      graph: { nodes: outNodes, edges: outEdges },
    },
  };
}
