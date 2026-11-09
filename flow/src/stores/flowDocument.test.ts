/* Task M2.3: export/import unit tests. The required cases target the
 * pure functions in ./stores/flowDocument directly; the store-loading
 * case pins the setGraph wiring without a DOM (the Blob download and
 * file picker are browser-only and stay thin in App.tsx). */
import { describe, it, expect, beforeEach } from "vitest";
import { serializeFlow, parseFlowDocument, validateFlowDocument } from "./flowDocument";
import { useFlowStore } from "../store";
import type { Graph } from "../store";

const graph: Graph = {
  nodes: [
    { id: "n1", kind: "dataset", label: "RowTour-360", props: {}, position: { x: 0, y: 0 } },
    { id: "n2", kind: "train", props: { steps: 4 }, position: { x: 120, y: 0 } },
  ],
  edges: [{ id: "e1", from: "n1", to: "n2", fromPort: "cleaned", toPort: "shard-dir" }],
};

describe("serializeFlow (export produces valid flow/0.1 JSON)", () => {
  it("serializes a graph that round-trips through the parser", () => {
    const text = serializeFlow("my-run", graph);
    const parsed = JSON.parse(text);
    expect(parsed.schema).toBe("flow/0.1");
    expect(parsed.meta).toEqual({ name: "my-run" });
    expect(parsed.graph).toEqual(graph);
    const res = parseFlowDocument(text);
    expect(res.error).toBeUndefined();
    expect(res.doc).toBeDefined();
    expect(res.doc?.schema).toBe("flow/0.1");
    expect(res.doc?.meta).toEqual({ name: "my-run" });
    expect(res.doc?.graph).toEqual(graph);
  });

  it("emits pretty-printed JSON (2-space indent)", () => {
    const text = serializeFlow("x", { nodes: [], edges: [] });
    expect(text).toBe(
      JSON.stringify(
        { schema: "flow/0.1", meta: { name: "x" }, graph: { nodes: [], edges: [] } },
        null,
        2,
      ),
    );
  });
});

describe("parseFlowDocument (import parses and loads)", () => {
  it("parses a valid document and returns the doc", () => {
    const res = parseFlowDocument(serializeFlow("import-me", graph));
    expect(res.error).toBeUndefined();
    expect(res.doc?.meta.name).toBe("import-me");
    expect(res.doc?.graph.nodes).toEqual(graph.nodes);
    expect(res.doc?.graph.edges).toEqual(graph.edges);
  });

  it("loads the parsed graph into the store via setGraph", () => {
    useFlowStore.setState({
      graph: { nodes: [], edges: [] },
      past: [],
      future: [],
      projection: [],
      validatedDoc: null,
      error: null,
    });
    const res = parseFlowDocument(serializeFlow("import-me", graph));
    expect(res.error).toBeUndefined();
    useFlowStore.getState().setGraph(res.doc!.graph);
    expect(useFlowStore.getState().graph).toEqual(graph);
  });

  it("normalizes stray fields so only domain fields reach the store", () => {
    const messy = {
      schema: "flow/0.1",
      meta: { name: "messy", extra: true },
      graph: {
        nodes: [
          {
            id: "n1",
            kind: "dataset",
            props: {},
            position: { x: 1, y: 2 },
            measured: { width: 10, height: 10 },
            dragging: true,
          },
        ],
        edges: [{ id: "e1", from: "n1", to: "n1", fromPort: "a", toPort: "b", selected: true }],
      },
    };
    const res = validateFlowDocument(messy);
    expect(res.error).toBeUndefined();
    expect(res.doc?.graph.nodes[0]).toEqual({
      id: "n1",
      kind: "dataset",
      props: {},
      position: { x: 1, y: 2 },
    });
    expect(res.doc?.graph.edges[0]).toEqual({
      id: "e1",
      from: "n1",
      to: "n1",
      fromPort: "a",
      toPort: "b",
    });
  });
});

describe("parseFlowDocument rejects invalid documents", () => {
  beforeEach(() => {
    useFlowStore.setState({ graph: { nodes: [], edges: [] }, error: null });
  });

  it("rejects malformed JSON with an error result and no doc", () => {
    const res = parseFlowDocument("{not json");
    expect(res.error).toBeDefined();
    expect(res.error).toContain("invalid JSON");
    expect(res.doc).toBeUndefined();
  });

  it("rejects a non-object document", () => {
    for (const text of ["[1,2]", '"flow/0.1"', "42", "null"]) {
      const res = parseFlowDocument(text);
      expect(res.error).toBeDefined();
      expect(res.doc).toBeUndefined();
    }
  });

  it("rejects a wrong schema tag", () => {
    const res = validateFlowDocument({
      schema: "flow/0.2",
      meta: { name: "x" },
      graph: { nodes: [], edges: [] },
    });
    expect(res.error).toMatch(/schema/);
    expect(res.doc).toBeUndefined();
  });

  it("rejects missing meta / meta.name", () => {
    const noMeta = validateFlowDocument({ schema: "flow/0.1", graph: { nodes: [], edges: [] } });
    expect(noMeta.error).toMatch(/meta/);
    const noName = validateFlowDocument({ schema: "flow/0.1", meta: {}, graph: { nodes: [], edges: [] } });
    expect(noName.error).toMatch(/meta\.name/);
  });

  it("rejects missing or mistyped graph fields", () => {
    const noGraph = validateFlowDocument({ schema: "flow/0.1", meta: { name: "x" } });
    expect(noGraph.error).toMatch(/graph/);
    const nodesNotArray = validateFlowDocument({
      schema: "flow/0.1",
      meta: { name: "x" },
      graph: { nodes: "nope", edges: [] },
    });
    expect(nodesNotArray.error).toMatch(/nodes/);
    const edgesNotArray = validateFlowDocument({
      schema: "flow/0.1",
      meta: { name: "x" },
      graph: { nodes: [], edges: {} },
    });
    expect(edgesNotArray.error).toMatch(/edges/);
  });

  it("rejects nodes/edges with missing or mistyped fields", () => {
    const base = { schema: "flow/0.1", meta: { name: "x" } };
    const noProps = validateFlowDocument({
      ...base,
      graph: { nodes: [{ id: "n1", kind: "dataset" }], edges: [] },
    });
    expect(noProps.error).toMatch(/props/);
    const badPos = validateFlowDocument({
      ...base,
      graph: { nodes: [{ id: "n1", kind: "dataset", props: {}, position: { x: "0", y: 0 } }], edges: [] },
    });
    expect(badPos.error).toMatch(/position/);
    const badEdge = validateFlowDocument({
      ...base,
      graph: { nodes: [], edges: [{ id: "e1", from: "n1", to: "n2", fromPort: "a" }] },
    });
    expect(badEdge.error).toMatch(/toPort/);
  });

  it("a rejection toasts the error and never loads the graph", () => {
    const before = useFlowStore.getState().graph;
    const res = parseFlowDocument("{not json");
    expect(res.error).toBeDefined();
    // Toolbar wiring: the parser's error result IS the toast text.
    useFlowStore.getState().setError("import failed: " + res.error);
    const st = useFlowStore.getState();
    expect(st.error).toContain("import failed");
    expect(st.error).toContain("invalid JSON");
    expect(st.graph).toEqual(before); // no load on rejection
  });
});
