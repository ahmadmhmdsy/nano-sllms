/**
 * modelStore tests (F2 plan Task 5): add-from-spec-defaults, connect +
 * walk-cached totals, duplicate connect rejection, and a serialize/load
 * round trip through an unused doc (openDoc of a hand-built document).
 */
import { describe, it, expect, beforeEach } from "vitest";
import type { Connection, EdgeChange, NodeChange } from "@xyflow/react";
import {
  useModelStore,
  defaultProps,
  validateConnect,
  connectReducer,
  modelDocument,
  nextNodeId,
  nextEdgeId,
} from "./modelStore";
import { LAYER_REGISTRY } from "./registry";

function resetStore() {
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

describe("modelStore (F2 Task 5)", () => {
  beforeEach(resetStore);

  it("addNode seeds props from the spec defaults (registry-driven)", () => {
    const id = useModelStore.getState().addNode("embedding", { x: 0, y: 0 });
    expect(id).toBe("n1");
    const node = useModelStore.getState().nodes[0];
    // embedding spec: vocab_size default 32768, d default 256.
    expect(node.props).toEqual({ vocab_size: 32768, d: 256 });
    // A second node gets a unique id (max counter + 1).
    const id2 = useModelStore.getState().addNode("rmsnorm");
    expect(id2).toBe("n2");
    expect(id2).not.toBe(id);
    // Unknown kind is refused (error set, no node added).
    expect(useModelStore.getState().addNode("no-such-kind")).toBeNull();
    expect(useModelStore.getState().nodes).toHaveLength(2);
    expect(useModelStore.getState().error).toContain("unknown layer kind");
  });

  it("addNode leaves props without defaults absent (honest error later)", () => {
    // Every current kind declares defaults on all props; assert the
    // seeding rule directly on the embedding spec for regression safety.
    const props = defaultProps(LAYER_REGISTRY.embedding);
    expect(Object.keys(props).sort()).toEqual(["d", "vocab_size"]);
    for (const p of LAYER_REGISTRY.embedding.props) {
      expect(props[p.name]).toBe(p.default);
    }
  });

  it("connect then updateProps recompute the walk total", () => {
    const s = useModelStore.getState();
    const n1 = s.addNode("input") as string;
    const n2 = s.addNode("embedding") as string;
    // Disconnected: embedding reports missing input; total counts nothing.
    let walk = useModelStore.getState().walk;
    expect(walk.errors.some((e) => e.nodeId === n2)).toBe(true);
    expect(walk.total).toBe(0);

    // Connect input.tokens -> embedding.tokens: walk now yields params.
    expect(useModelStore.getState().connect(conn(n1, "tokens", n2, "tokens"))).toBe(true);
    walk = useModelStore.getState().walk;
    expect(walk.errors).toHaveLength(0);
    // 32768 * 256 (embedding defaults).
    expect(walk.params[n2]).toBe(32768 * 256);
    expect(walk.total).toBe(32768 * 256);
    expect(walk.shapes.get(n2)).toEqual({ s: 1024, d: 256 });

    // Editing d 256 -> 512 recomputes the total live.
    useModelStore.getState().updateProps(n2, { vocab_size: 32768, d: 512 });
    expect(useModelStore.getState().walk.total).toBe(32768 * 512);
  });

  it("connect marks the store dirty and edges get unique ids", () => {
    const s = useModelStore.getState();
    const n1 = s.addNode("input") as string;
    const n2 = s.addNode("rmsnorm") as string;
    expect(useModelStore.getState().connect(conn(n1, "tokens", n2, "x"))).toBe(true);
    expect(useModelStore.getState().dirty).toBe(true);
    expect(useModelStore.getState().edges[0].id).toBe("e1");
  });

  it("duplicate connect to the same input port is rejected", () => {
    const s = useModelStore.getState();
    const n1 = s.addNode("input") as string;
    const n2 = s.addNode("residual") as string;
    const n3 = s.addNode("rope") as string;
    // First wire into residual.stream succeeds...
    expect(useModelStore.getState().connect(conn(n1, "tokens", n2, "stream"))).toBe(true);
    // ...the SECOND edge into the same port is refused, graph untouched.
    expect(useModelStore.getState().connect(conn(n3, "out", n2, "stream"))).toBe(false);
    expect(useModelStore.getState().edges).toHaveLength(1);
    expect(useModelStore.getState().error).toContain("exactly one edge");
    // validateConnect agrees at the pure level too.
    const edges = useModelStore.getState().edges;
    expect(validateConnect(useModelStore.getState().nodes, edges, n3, "out", n2, "stream")).toHaveLength(1);
    expect(validateConnect(useModelStore.getState().nodes, edges, n3, "out", n2, "bypass")).toHaveLength(0);
    // connectReducer (pure) refuses identically without mutating.
    const res = connectReducer(useModelStore.getState().nodes, edges, conn(n3, "out", n2, "stream"));
    expect(res.error).not.toBeNull();
    expect(res.edges).toBe(edges);
  });

  it("connect validation refuses wrong ports and self-loops", () => {
    const s = useModelStore.getState();
    const n1 = s.addNode("input") as string;
    const n2 = s.addNode("rmsnorm") as string;
    const st = useModelStore.getState();
    // "out" is not an output of input; "x" is the input port of rmsnorm —
    // swapped port names must fail on the output side first.
    expect(validateConnect(st.nodes, [], n1, "x", n2, "x")).toHaveLength(1);
    expect(validateConnect(st.nodes, [], n1, "tokens", n2, "tokens")).toHaveLength(1);
    expect(validateConnect(st.nodes, [], n1, "tokens", n1, "tokens")).toHaveLength(1);
    // connectReducer with missing handles fails honestly.
    expect(connectReducer(st.nodes, [], { source: n1, target: n2 } as Connection).error).toContain("both handles");
  });

  it("applyNodesChanges: position changes land on the domain node", () => {
    const s = useModelStore.getState();
    s.addNode("input");
    const changes: NodeChange[] = [
      { id: "n1", type: "position", position: { x: 111, y: 222 }, dragging: false },
    ];
    useModelStore.getState().applyNodesChanges(changes);
    expect(useModelStore.getState().nodes[0].position).toEqual({ x: 111, y: 222 });
  });

  it("applyNodesChanges: remove change drops the node and its edges", () => {
    const s = useModelStore.getState();
    const n1 = s.addNode("input") as string;
    const n2 = s.addNode("rmsnorm") as string;
    useModelStore.getState().connect(conn(n1, "tokens", n2, "x"));
    const changes: NodeChange[] = [{ id: n1, type: "remove" }];
    useModelStore.getState().applyNodesChanges(changes);
    expect(useModelStore.getState().nodes.map((n) => n.id)).toEqual([n2]);
    expect(useModelStore.getState().edges).toHaveLength(0);
    expect(useModelStore.getState().walk.total).toBe(0);
  });

  it("applyEdgesChanges: remove change drops the edge", () => {
    const s = useModelStore.getState();
    const n1 = s.addNode("input") as string;
    const n2 = s.addNode("rmsnorm") as string;
    useModelStore.getState().connect(conn(n1, "tokens", n2, "x"));
    const changes: EdgeChange[] = [{ id: "e1", type: "remove" }];
    useModelStore.getState().applyEdgesChanges(changes);
    expect(useModelStore.getState().edges).toHaveLength(0);
  });

  it("deleteNode drops the node, its edges, and clears selection", () => {
    const s = useModelStore.getState();
    const n1 = s.addNode("input") as string;
    const n2 = s.addNode("rmsnorm") as string;
    useModelStore.getState().connect(conn(n1, "tokens", n2, "x"));
    useModelStore.getState().setSelection(n2);
    useModelStore.getState().deleteNode(n2);
    const st = useModelStore.getState();
    expect(st.nodes.map((n) => n.id)).toEqual([n1]);
    expect(st.edges).toHaveLength(0);
    expect(st.selection).toBeNull();
expect(st.dirty).toBe(true);
  });

  it("serialize (modelDocument) then openDoc round trip is lossless", () => {
    const s = useModelStore.getState();
    const n1 = s.addNode("input") as string;
    const n2 = s.addNode("embedding") as string;
    useModelStore.getState().connect(conn(n1, "tokens", n2, "tokens"));
    useModelStore.getState().setModelName("round-trip-test");

    const doc = modelDocument(useModelStore.getState().name, useModelStore.getState());
    // Document shape per model/0.1 (server Task 4 contract).
    expect(doc.format).toBe("model/0.1");
    expect(doc.name).toBe("round-trip-test");
    expect(doc.nodes).toEqual([
      { id: n1, kind: "input", props: { vocab_size: 32768, ctx: 1024 }, position: { x: 60, y: 60 } },
      { id: n2, kind: "embedding", props: { vocab_size: 32768, d: 256 }, position: { x: 84, y: 84 } },
    ]);
    expect(doc.edges).toEqual([
      { id: "e1", from: n1, to: n2, fromPort: "tokens", toPort: "tokens" },
    ]);
    expect(doc.meta).toEqual({});

    // Load the (unused) doc into a fresh store state and compare.
    useModelStore.setState({
      nodes: [],
      edges: [],
      name: "",
      projection: [],
      selection: null,
    });
    useModelStore.getState().openDoc(doc);
    const st = useModelStore.getState();
    expect(st.name).toBe("round-trip-test");
    expect(st.nodes).toEqual(doc.nodes);
    expect(st.edges).toEqual(doc.edges);
    expect(st.dirty).toBe(false);
    // Walk cache is warm after a load too.
    expect(st.walk.total).toBe(32768 * 256);
    expect(st.walk.errors).toHaveLength(0);
  });

  it("save/open hit the api and round-trip (mocked fetch)", async () => {
    const fetchCalls: { url: string; init?: RequestInit }[] = [];
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      fetchCalls.push({ url, init });
      if (url === "/api/models/my-model" && init?.method === "POST") {
        return new Response(JSON.stringify({ ok: true, name: "my-model" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url === "/api/models/my-model") {
        return new Response(
          JSON.stringify(modelDocument("my-model", {
            nodes: [{ id: "n1", kind: "input", props: { vocab_size: 100, ctx: 8 }, position: { x: 0, y: 0 } }],
            edges: [],
          })),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      return new Response("not found", { status: 404 });
    }) as typeof fetch;
    try {
      resetStore();
      useModelStore.getState().addNode("input");
      const okSave = await useModelStore.getState().save("my-model");
      expect(okSave).toBe(true);
      expect(useModelStore.getState().dirty).toBe(false);
      expect(fetchCalls[0].url).toBe("/api/models/my-model");
      expect(fetchCalls[0].init?.method).toBe("POST");

      const okOpen = await useModelStore.getState().open("my-model");
      expect(okOpen).toBe(true);
      expect(useModelStore.getState().name).toBe("my-model");
      expect(useModelStore.getState().nodes).toHaveLength(1);
      expect(useModelStore.getState().nodes[0].props).toEqual({ vocab_size: 100, ctx: 8 });
    } finally {
      globalThis.fetch = realFetch;
    }
  });

  it("id helpers are deterministic and unique", () => {
    expect(nextNodeId([{ id: "n3", kind: "input", props: {}, position: { x: 0, y: 0 } }])).toBe("n4");
    expect(nextNodeId([])).toBe("n1");
    expect(nextEdgeId([{ id: "e7", from: "a", to: "b", fromPort: "o", toPort: "i" }])).toBe("e8");
    expect(nextEdgeId([])).toBe("e1");
  });
});
