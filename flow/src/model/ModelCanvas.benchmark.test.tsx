/**
 * Task M6.1 render benchmark — model canvas (ModelCanvas), plus the
 * walk-memoization verification for the same task.
 *
 * Same design as FlowCanvas.benchmark.test.tsx: a 100-node model graph
 * (input -> embedding -> 98 rmsnorm, all ports valid, grid layout) is
 * projected through toXYNodes (the exact projection ModelCanvas feeds
 * to ReactFlow) and the 100 ModelNode components must render in
 * < 100ms, median of 7 runs after a warmup. See the FlowCanvas benchmark
 * header for why the node projection is timed instead of a controlled
 * <ReactFlow> SSR render (React Flow v12 syncs controlled nodes from
 * an effect; renderToStaticMarkup runs no effects).
 *
 * The two additional tests verify the walk side of the task:
 * walkModelGraph stays fast on a large graph, and the store memoizes
 * the walk on graph object identity (a select-only change reuses the
 * cached walk object instead of recomputing the topological walk).
 */
import { describe, it, expect, beforeEach } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { Connection } from "@xyflow/react";
import { ReactFlowProvider, type NodeProps } from "@xyflow/react";
import ModelCanvas from "./ModelCanvas";
import { useModelStore, toXYNodes, modelDocument } from "./modelStore";
import type { DomainEdge, DomainNode } from "../stores/graphStore";
import { walkModelGraph } from "./graphWalk";
import { LAYER_REGISTRY } from "./registry";
import ModelNode from "./ModelNode";
import type { ModelNode as ModelNodeType } from "./modelStore";

const NODE_COUNT = 100;
const RUNS = 7;
const RENDER_THRESHOLD_MS = 100;
const WALK_THRESHOLD_MS = 20;

/** input -> embedding -> (n-2) rmsnorm chain — every port valid per
 * LAYER_REGISTRY, so the walk completes with zero errors. */
function modelGraph(n: number) {
  // Typed as the domain document graph (props: Record<string, unknown>)
  // so the per-kind prop shapes below check against the domain rather
  // than a literal-union inference.
  const nodes: DomainNode[] = [
    { id: "n1", kind: "input", props: { vocab_size: 32768, ctx: 1024 }, position: { x: 0, y: 0 } },
    { id: "n2", kind: "embedding", props: { vocab_size: 32768, d: 256 }, position: { x: 220, y: 0 } },
  ];
  const edges: DomainEdge[] = [{ id: "e1", from: "n1", to: "n2", fromPort: "tokens", toPort: "tokens" }];
  for (let i = 2; i < n; i++) {
    const id = "n" + (i + 1);
    nodes.push({ id, kind: "rmsnorm", props: {}, position: { x: (i % 10) * 220, y: Math.floor(i / 10) * 120 } });
    edges.push({ id: "e" + (i + 1), from: "n" + i, to: id, fromPort: "out", toPort: "x" });
  }
  return { nodes, edges };
}

/** Hand-rolled NodeProps (same pattern as ModelNode.test.tsx). */
function nodeProps(n: ModelNodeType): NodeProps<ModelNodeType> {
  return {
    id: n.id,
    type: "model",
    data: n.data,
    selected: false,
    dragging: false,
    zIndex: 0,
    selectable: true,
    deletable: true,
    draggable: true,
    isConnectable: true,
    positionAbsoluteX: n.position.x,
    positionAbsoluteY: n.position.y,
  };
}

function median(nums: number[]): number {
  const sorted = [...nums].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function resetModelStore() {
  useModelStore.setState({
    nodes: [],
    edges: [],
    name: "",
    dirty: false,
    walk: { shapes: new Map(), params: {}, total: 0, errors: [] },
    projection: [],
    selection: null,
    error: null,
  });
}

const conn = (from: string, fromPort: string, to: string, toPort: string): Connection => ({
  source: from,
  sourceHandle: fromPort,
  target: to,
  targetHandle: toPort,
});

describe("Task M6.1 ModelCanvas render benchmark", () => {
  beforeEach(resetModelStore);

  it("renders a 100-node model graph in < 100ms (median of 7 runs)", () => {
    const graph = modelGraph(NODE_COUNT);
    // openDoc is the realistic load path: it computes the walk once.
    useModelStore.getState().openDoc(modelDocument("bench", graph));
    const { nodes, edges, walk } = useModelStore.getState();
    // The exact projection ModelCanvas feeds to ReactFlow.
    const xyNodes = toXYNodes({ nodes, edges }, walk);

    const render = () =>
      renderToStaticMarkup(
        <ReactFlowProvider>
          {xyNodes.map((n) => (
            <ModelNode key={n.id} {...nodeProps(n)} />
          ))}
        </ReactFlowProvider>,
      );

    const html = render(); // warmup (JIT) + content sanity check
    const renderedNodes = html.split('class="model-node"').length - 1;
    expect(renderedNodes).toBe(NODE_COUNT);

    const times: number[] = [];
    for (let i = 0; i < RUNS; i++) {
      const t0 = performance.now();
      render();
      times.push(performance.now() - t0);
    }
    const med = median(times);
    console.log(
      "[bench] ModelCanvas " + NODE_COUNT + " nodes: median=" + med.toFixed(2) +
      "ms min=" + Math.min(...times).toFixed(2) + "ms max=" + Math.max(...times).toFixed(2) +
      "ms (threshold " + RENDER_THRESHOLD_MS + "ms, " + RUNS + " runs)",
    );
    expect(med).toBeLessThan(RENDER_THRESHOLD_MS);
  });

  it("the full canvas mounts with a 100-node model graph (smoke)", () => {
    const graph = modelGraph(NODE_COUNT);
    useModelStore.getState().openDoc(modelDocument("bench-smoke", graph));
    const html = renderToStaticMarkup(<ModelCanvas />);
    expect(html).toContain("model-header");
    expect(html).toContain("react-flow");
  });

  it("walkModelGraph walks a 100-node graph in < 20ms (median of 10 runs)", () => {
    const graph = modelGraph(NODE_COUNT);
    const times: number[] = [];
    let walk;
    for (let i = 0; i < 10; i++) {
      const t0 = performance.now();
      walk = walkModelGraph(graph, LAYER_REGISTRY);
      times.push(performance.now() - t0);
    }
    expect(walk!.errors).toHaveLength(0);
    const med = median(times);
    console.log(
      "[bench] walkModelGraph " + NODE_COUNT + " nodes: median=" + med.toFixed(2) +
      "ms min=" + Math.min(...times).toFixed(2) + "ms max=" + Math.max(...times).toFixed(2) +
      "ms (threshold " + WALK_THRESHOLD_MS + "ms, 10 runs)",
    );
    expect(med).toBeLessThan(WALK_THRESHOLD_MS);
  });

  it("applyNodesChanges reuses the cached walk for select-only changes (identity memoization)", () => {
    const s = useModelStore.getState();
    const n1 = s.addNode("input") as string;
    const n2 = s.addNode("rmsnorm") as string;
    expect(useModelStore.getState().connect(conn(n1, "tokens", n2, "x"))).toBe(true);
    const walkBefore = useModelStore.getState().walk;

    // Select-only change: round-trips a NEW graph object whose node/edge
    // entries are all identical references — the walk must be reused.
    useModelStore.getState().applyNodesChanges([{ id: n1, type: "select", selected: true }]);
    expect(useModelStore.getState().selection).toBe(n1);
    expect(useModelStore.getState().walk).toBe(walkBefore);

    // Position change: node identity changes — the walk recomputes.
    useModelStore.getState().applyNodesChanges([
      { id: n1, type: "position", position: { x: 42, y: 42 }, dragging: false },
    ]);
    expect(useModelStore.getState().nodes[0].position).toEqual({ x: 42, y: 42 });
    expect(useModelStore.getState().walk).not.toBe(walkBefore);
  });
});
