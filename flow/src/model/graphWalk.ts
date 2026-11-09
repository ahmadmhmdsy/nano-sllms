/**
 * Topological shape inference + parameter rollup over a model graph
 * (F2 plan Task 3).
 *
 * Kahn walk over the model/0.1 document graph. DomainNode/DomainEdge are
 * REUSED from stores/graphStore (same node/edge field shape as the
 * pipeline graph — one declaration, not two), via type-only imports so
 * nothing runtime-heavy is pulled into this pure module.
 *
 * Honest-error policy: per-node errors never abort the walk — every node
 * is judged on its own, so other components still get shapes/params.
 * `total` sums paramCount over nodes WITHOUT any recorded error.
 */
import type { DomainEdge, DomainNode } from "../stores/graphStore";
import type { LayerSpec, Shape } from "./layers/shared";

/** Re-exported so consumers of the walker need no registry import. */
export type { Shape } from "./layers/shared";

export interface WalkError {
  nodeId: string;
  message: string;
}

export interface WalkResult {
  /** Inferred output shape per successfully walked node. */
  shapes: Map<string, Shape>;
  /** paramCount per walked node (present only when the node has no errors). */
  params: Record<string, number>;
  /** Sum of params over nodes WITHOUT errors. */
  total: number;
  errors: WalkError[];
}

/** Structural graph input: the model/0.1 document graph (DomainNode adds
 * label/position, DomainEdge matches exactly — both accepted as-is). */
export interface WalkGraph {
  nodes: Pick<DomainNode, "id" | "kind" | "props">[];
  edges: DomainEdge[];
}

/** Kahn topological walk with per-node shape inference + param rollup. */
export function walkModelGraph(
  graph: WalkGraph,
  registry: Record<string, LayerSpec>,
): WalkResult {
  const errors: WalkError[] = [];
  const shapes = new Map<string, Shape>();
  const params: Record<string, number> = {};

  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));

  // ---- Edge registration -------------------------------------------------
  // Incoming edges per target node (in document order). Edges with missing
  // endpoints are honest errors on their existing endpoint — a dangling
  // wire can never deliver a shape — and never enter the walk.
  const incoming = new Map<string, DomainEdge[]>();
  const outgoing = new Map<string, DomainEdge[]>();
  const push = (map: Map<string, DomainEdge[]>, key: string, e: DomainEdge) => {
    const list = map.get(key);
    if (list) list.push(e);
    else map.set(key, [e]);
  };
  for (const e of graph.edges) {
    if (!nodeById.has(e.from) && !nodeById.has(e.to)) continue;
    if (!nodeById.has(e.from)) {
      errors.push({
        nodeId: e.to,
        message: `edge '${e.id}': source node '${e.from}' does not exist`,
      });
      continue;
    }
    if (!nodeById.has(e.to)) {
      errors.push({
        nodeId: e.from,
        message: `edge '${e.id}': target node '${e.to}' does not exist`,
      });
      continue;
    }
    push(incoming, e.to, e);
    push(outgoing, e.from, e);
  }

  // Kahn indegree per registered edge.
  const indegree = new Map<string, number>();
  for (const n of graph.nodes) indegree.set(n.id, 0);
  for (const [, list] of incoming) {
    for (const e of list) indegree.set(e.to, (indegree.get(e.to) ?? 0) + 1);
  }

  // ---- Per-node processing ------------------------------------------------
  const visited = new Set<string>();

  const processNode = (node: Pick<DomainNode, "id" | "kind" | "props">): void => {
    visited.add(node.id);
    const spec = registry[node.kind];
    if (!spec) {
      errors.push({ nodeId: node.id, message: `unknown kind '${node.kind}'` });
      return; // no shape propagates; downstream reports unavailable inputs
    }

    // Group incoming edges per declared input port (document order). Model
    // edges must NAME ports: an edge to a port that does not exist on the
    // target kind is an honest error and binds to nothing.
    const byPort = new Map<string, DomainEdge[]>();
    let failed = false;
    for (const e of incoming.get(node.id) ?? []) {
      if (!spec.inputs.some((p) => p.name === e.toPort)) {
        errors.push({
          nodeId: node.id,
          message: `edge '${e.id}': '${e.toPort}' is not an input port of kind '${node.kind}'`,
        });
        continue;
      }
      push(byPort, e.toPort, e);
    }

    // Gather input shapes in spec port order; enforce honesty per port.
    const inputs: Shape[] = [];
    for (const port of spec.inputs) {
      const portEdges = byPort.get(port.name) ?? [];
      if (portEdges.length === 0) {
        errors.push({ nodeId: node.id, message: `missing input ${port.name}` });
        failed = true;
        continue;
      }
      if (portEdges.length > 1) {
        errors.push({
          nodeId: node.id,
          message: `input port '${port.name}' has more than one incoming edge (${portEdges.map((e) => e.id).join(", ")})`,
        });
        failed = true;
        continue;
      }
      const e = portEdges[0];
      if (!shapes.has(e.from)) {
        errors.push({
          nodeId: node.id,
          message: `input '${port.name}' unavailable: upstream node '${e.from}' produced no shape`,
        });
        failed = true;
        continue;
      }
      inputs.push(shapes.get(e.from) as Shape);
    }
    // Outgoing edges must name a real output port of THIS kind — honest
    // error on the source node; the edge itself still delivers shape.
    for (const e of outgoing.get(node.id) ?? []) {
      if (!spec.outputs.some((p) => p.name === e.fromPort)) {
        errors.push({
          nodeId: node.id,
          message: `edge '${e.id}': '${e.fromPort}' is not an output port of kind '${node.kind}'`,
        });
      }
    }

    if (failed) return; // no shape/params for a structurally broken node

    // Registry functions may throw (prop validation, d mismatches, ...) —
    // caught and wrapped with the node id; the walk itself never aborts.
    let out: Shape;
    try {
      out = spec.inferShape(node.props, inputs);
    } catch (err) {
      errors.push({
        nodeId: node.id,
        message: `${node.id} (${node.kind}): ${err instanceof Error ? err.message : String(err)}`,
      });
      return;
    }
    shapes.set(node.id, out);
    try {
      params[node.id] = spec.paramCount(node.props, out, inputs);
    } catch (err) {
      errors.push({
        nodeId: node.id,
        message: `${node.id} (${node.kind}): ${err instanceof Error ? err.message : String(err)}`,
      });
      delete params[node.id]; // errored node never contributes to total
    }
  };

  // ---- Kahn main loop -----------------------------------------------------
  const queue: string[] = graph.nodes
    .filter((n) => (indegree.get(n.id) ?? 0) === 0)
    .map((n) => n.id);
  while (queue.length > 0) {
    const id = queue.shift() as string;
    const node = nodeById.get(id);
    if (node) processNode(node);
    // Decrement EVERY outgoing edge — even when this node failed — so
    // downstream nodes still get judged (their own honest error) instead
    // of hanging behind a broken upstream.
    for (const e of outgoing.get(id) ?? []) {
      const next = (indegree.get(e.to) ?? 0) - 1;
      indegree.set(e.to, next);
      if (next === 0) queue.push(e.to);
    }
  }

  // ---- Cycle handling -----------------------------------------------------
  // Nodes never reached by Kahn are on a cycle or downstream of one.
  const unreachable = graph.nodes.filter((n) => !visited.has(n.id));

  // Cycle members: reachable from themselves via edges among the unreachable.
  const unIds = new Set(unreachable.map((n) => n.id));
  const cycleMembers = new Set<string>();
  for (const n of unreachable) {
    const seen = new Set<string>();
    const stack = (outgoing.get(n.id) ?? [])
      .map((e) => e.to)
      .filter((to) => unIds.has(to));
    let onCycle = false;
    while (stack.length > 0 && !onCycle) {
      const cur = stack.pop() as string;
      if (cur === n.id) {
        onCycle = true;
        break;
      }
      if (seen.has(cur)) continue;
      seen.add(cur);
      for (const e of outgoing.get(cur) ?? []) {
        if (unIds.has(e.to)) stack.push(e.to);
      }
    }
    if (onCycle) cycleMembers.add(n.id);
  }
  for (const n of unreachable) {
    if (cycleMembers.has(n.id)) {
      visited.add(n.id);
      errors.push({ nodeId: n.id, message: "cycle detected" });
    }
  }

  // Non-cycle unreachable nodes are merely downstream of a failed/cycle
  // node: keep judging them (in dependency order) so they get their own
  // honest "unavailable input" errors instead of a silent gap.
  let progressed = true;
  while (progressed) {
    progressed = false;
    for (const n of unreachable) {
      if (visited.has(n.id)) continue;
      const deps = (incoming.get(n.id) ?? []).map((e) => e.from);
      if (!deps.every((d) => visited.has(d))) continue;
      processNode(n);
      progressed = true;
    }
  }
  // Defensive tail: anything still unvisited is reported as cycle-blocked
  // rather than silently dropped.
  for (const n of unreachable) {
    if (!visited.has(n.id)) errors.push({ nodeId: n.id, message: "cycle detected" });
  }

  // ---- Rollup -------------------------------------------------------------
  let total = 0;
  for (const id of Object.keys(params)) total += params[id];
  return { shapes, params, total, errors };
}
