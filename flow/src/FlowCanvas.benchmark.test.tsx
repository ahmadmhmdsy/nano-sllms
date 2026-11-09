/**
 * Task M6.1 render benchmark — pipeline canvas (FlowCanvas).
 *
 * A 100-node pipeline graph (10x10 grid, so most nodes sit outside any
 * single viewport) must render in < 100ms.
 *
 * WHAT IS TIMED: the node projection — toXYNodes(graph, registry) fed
 * through the same PipelineNode components React Flow renders per node.
 * This is the render cost that scales with graph size (the frame-drop
 * source on pan/zoom); the canvas chrome (pane, Controls, Background,
 * MiniMap) is node-count-independent.
 *
 * WHY NOT renderToStaticMarkup(<FlowCanvas />): React Flow v12 syncs
 * the controlled `nodes` prop into its store from a StoreUpdater
 * effect, and renderToStaticMarkup runs NO effects — a controlled
 * ReactFlow renders zero nodes under SSR (verified: the nodes pane is
 * empty with or without onlyRenderVisibleElements). The repo has no
 * jsdom/happy-dom installed, so a DOM-hydrated render is not possible
 * without new dependencies. Rendering the projected node components
 * directly follows the established repo pattern (ModelNode.test.tsx)
 * and exercises the exact per-node render path.
 *
 * RESILIENCE: performance.now() around renderToStaticMarkup, median of 7
 * measured runs after one warmup render. The median (not the mean)
 * keeps a single GC/scheduler outlier from failing a healthy run; the
 * 100ms threshold is ~10x the measured cost, so the test catches real
 * regressions (e.g. an accidental O(N^2) in the projection) without
 * being flaky on a slow CI machine.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ReactFlowProvider, type NodeProps } from "@xyflow/react";
import FlowCanvas from "./FlowCanvas";
import { useFlowStore, toXYNodes, FALLBACK_REGISTRY } from "./store";
import PipelineNode from "./nodes/PipelineNode";
import type { PipelineNode as PipelineNodeType } from "./store";

const NODE_COUNT = 100;
const RUNS = 7;
const THRESHOLD_MS = 100;

/** Chain of the six fallback kinds, consecutive ports wired. */
const PIPELINE_KINDS = ["dataset", "prepare", "tokenize", "train", "eval", "infer"] as const;
const PIPELINE_OUT: Record<string, string> = {
  dataset: "cleaned",
  prepare: "cleaned-dir",
  tokenize: "shard-dir",
  train: "ckpt-dir",
  eval: "report",
  infer: "ckpt-dir",
};
const PIPELINE_IN: Record<string, string> = {
  dataset: "raw-dir",
  prepare: "raw-dir",
  tokenize: "cleaned-dir",
  train: "shard-dir",
  eval: "ckpt-dir",
  infer: "ckpt-dir",
};

/** n nodes in a chain, laid out on a 10-column grid (the viewport-
 * culling workload: any single viewport sees only a fraction). */
function pipelineGraph(n: number) {
  const nodes = [];
  const edges = [];
  for (let i = 0; i < n; i++) {
    const kind = PIPELINE_KINDS[i % PIPELINE_KINDS.length];
    nodes.push({
      id: "n" + (i + 1),
      kind,
      props: {},
      position: { x: (i % 10) * 240, y: Math.floor(i / 10) * 140 },
    });
    if (i > 0) {
      const prev = nodes[i - 1];
      edges.push({
        id: "e" + i,
        from: prev.id,
        to: "n" + (i + 1),
        fromPort: PIPELINE_OUT[prev.kind],
        toPort: PIPELINE_IN[kind],
      });
    }
  }
  return { nodes, edges };
}

/** Hand-rolled NodeProps — xyflow v12 requires type/dragging/zIndex/
 * selectable/deletable/draggable/isConnectable/positionAbsoluteX/Y on
 * top of id/data (same pattern as ModelNode.test.tsx). */
function nodeProps(n: PipelineNodeType): NodeProps<PipelineNodeType> {
  return {
    id: n.id,
    type: "pipeline",
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

describe("Task M6.1 FlowCanvas render benchmark", () => {
  beforeEach(() => {
    useFlowStore.setState({
      graph: { nodes: [], edges: [] },
      projection: [],
      error: null,
      searchQuery: "",
      selection: null,
    });
  });

  it("renders a 100-node graph in < 100ms (median of 7 runs)", () => {
    const graph = pipelineGraph(NODE_COUNT);
    useFlowStore.setState({ graph, registry: FALLBACK_REGISTRY });
    // The exact projection FlowCanvas feeds to ReactFlow.
    const xyNodes = toXYNodes(graph, FALLBACK_REGISTRY, []);

    const render = () =>
      renderToStaticMarkup(
        <ReactFlowProvider>
          {xyNodes.map((n) => (
            <PipelineNode key={n.id} {...nodeProps(n)} />
          ))}
        </ReactFlowProvider>,
      );

    const html = render(); // warmup (JIT) + content sanity check
    const renderedNodes = html.split('class="pipeline-node"').length - 1;
    expect(renderedNodes).toBe(NODE_COUNT);

    const times: number[] = [];
    for (let i = 0; i < RUNS; i++) {
      const t0 = performance.now();
      render();
      times.push(performance.now() - t0);
    }
    const med = median(times);
    console.log(
      "[bench] FlowCanvas " + NODE_COUNT + " nodes: median=" + med.toFixed(2) +
      "ms min=" + Math.min(...times).toFixed(2) + "ms max=" + Math.max(...times).toFixed(2) +
      "ms (threshold " + THRESHOLD_MS + "ms, " + RUNS + " runs)",
    );
    expect(med).toBeLessThan(THRESHOLD_MS);
  });

  it("the full canvas mounts with a 100-node graph in the store (smoke)", () => {
    const graph = pipelineGraph(NODE_COUNT);
    useFlowStore.setState({ graph, registry: FALLBACK_REGISTRY });
    const html = renderToStaticMarkup(<FlowCanvas />);
    // Chrome-only under SSR (see file header): the canvas shell, the
    // MiniMap and the Controls must mount without throwing.
    expect(html).toContain("react-flow");
    expect(html).toContain("minimap");
    expect(html).toContain("react-flow__controls");
  });
});
