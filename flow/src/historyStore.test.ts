import { describe, it, expect, beforeEach } from "vitest";
import { useFlowStore } from "./store";
import type { RegistrySnapshot } from "./store";

const REGISTRY: RegistrySnapshot = {
  valid_kinds: ["dataset", "prepare", "tokenize", "train", "eval", "infer"],
  nodes: {
    dataset: {
      ports: [{ name: "cleaned", direction: "out", type: "raw-dir" }],
      props: [],
      numeric_props: [],
      gate: null,
      label_semantic: "hf-dataset-name",
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
      features: ["webui-chat-button"],
    },
  },
  gates: { dataset: null, prepare: null, tokenize: null, train: "gpu", eval: "gpu", infer: "gpu/cpu" },
};

describe("undo/redo history (M1.1)", () => {
  beforeEach(() => {
    useFlowStore.setState({
      graph: { nodes: [], edges: [] },
      projection: [],
      selection: null,
      inspectorTab: "properties",
      runStatus: null,
      paletteCollapsed: false,
      inspectorCollapsed: false,
      registry: REGISTRY,
      registryError: null,
      error: null,
      past: [],
      future: [],
    });
  });

  it("starts with empty history", () => {
    const s = useFlowStore.getState();
    expect(s.past).toEqual([]);
    expect(s.future).toEqual([]);
    expect(s.canUndo()).toBe(false);
    expect(s.canRedo()).toBe(false);
  });

  it("addNode pushes history and undo restores empty graph", () => {
    const s = useFlowStore.getState();
    s.addNode("dataset", { x: 10, y: 20 });
    expect(useFlowStore.getState().graph.nodes).toHaveLength(1);
    expect(useFlowStore.getState().past).toHaveLength(1);
    expect(useFlowStore.getState().canUndo()).toBe(true);

    useFlowStore.getState().undo();
    expect(useFlowStore.getState().graph.nodes).toHaveLength(0);
    expect(useFlowStore.getState().canUndo()).toBe(false);
    expect(useFlowStore.getState().canRedo()).toBe(true);
  });

  it("redo re-adds the node after undo", () => {
    const s = useFlowStore.getState();
    s.addNode("dataset", { x: 10, y: 20 });
    useFlowStore.getState().undo();
    useFlowStore.getState().redo();
    expect(useFlowStore.getState().graph.nodes).toHaveLength(1);
    expect(useFlowStore.getState().graph.nodes[0].kind).toBe("dataset");
    expect(useFlowStore.getState().canRedo()).toBe(false);
  });

  it("connect pushes history and undo removes the edge", () => {
    const s = useFlowStore.getState();
    s.addNode("dataset", { x: 0, y: 0 });
    s.addNode("prepare", { x: 100, y: 0 });
    // Clear history from adds to isolate connect
    useFlowStore.setState({ past: [], future: [] });

    const res = useFlowStore.getState().connect({
      source: "n1", target: "n2", sourceHandle: "cleaned", targetHandle: "raw-dir",
    });
    expect(res).toBe(true);
    expect(useFlowStore.getState().graph.edges).toHaveLength(1);
    expect(useFlowStore.getState().past).toHaveLength(1);

    useFlowStore.getState().undo();
    expect(useFlowStore.getState().graph.edges).toHaveLength(0);
    // Nodes should still be there (only the edge was undone)
    expect(useFlowStore.getState().graph.nodes).toHaveLength(2);
  });

  it("multiple undos/redos in sequence", () => {
    const s = useFlowStore.getState();
    s.addNode("dataset", { x: 0, y: 0 });
    s.addNode("prepare", { x: 100, y: 0 });
    s.addNode("tokenize", { x: 200, y: 0 });
    expect(useFlowStore.getState().graph.nodes).toHaveLength(3);
    expect(useFlowStore.getState().past).toHaveLength(3);

    useFlowStore.getState().undo();
    expect(useFlowStore.getState().graph.nodes).toHaveLength(2);
    useFlowStore.getState().undo();
    expect(useFlowStore.getState().graph.nodes).toHaveLength(1);
    useFlowStore.getState().undo();
    expect(useFlowStore.getState().graph.nodes).toHaveLength(0);
    expect(useFlowStore.getState().canUndo()).toBe(false);

    useFlowStore.getState().redo();
    expect(useFlowStore.getState().graph.nodes).toHaveLength(1);
    useFlowStore.getState().redo();
    useFlowStore.getState().redo();
    expect(useFlowStore.getState().graph.nodes).toHaveLength(3);
    expect(useFlowStore.getState().canRedo()).toBe(false);
  });

  it("new action after undo clears the future stack", () => {
    const s = useFlowStore.getState();
    s.addNode("dataset", { x: 0, y: 0 });
    s.addNode("prepare", { x: 100, y: 0 });
    useFlowStore.getState().undo();
    expect(useFlowStore.getState().canRedo()).toBe(true);

    // New action should clear future
    s.addNode("tokenize", { x: 200, y: 0 });
    expect(useFlowStore.getState().canRedo()).toBe(false);
    expect(useFlowStore.getState().graph.nodes).toHaveLength(2);
  });

  it("undo after props write-back restores old props", () => {
    const s = useFlowStore.getState();
    s.addNode("prepare", { x: 0, y: 0 });
    useFlowStore.setState({ past: [], future: [] });

    useFlowStore.getState().updateNodeProps("n1", { rows: 100 });
    expect(useFlowStore.getState().graph.nodes[0].props).toEqual({ rows: 100 });

    useFlowStore.getState().undo();
    expect(useFlowStore.getState().graph.nodes[0].props).toEqual({});
  });

  it("undo after label edit restores old label", () => {
    const s = useFlowStore.getState();
    s.addNode("dataset", { x: 0, y: 0 });
    useFlowStore.setState({ past: [], future: [] });

    useFlowStore.getState().updateNodeLabel("n1", "my-dataset");
    expect(useFlowStore.getState().graph.nodes[0].label).toBe("my-dataset");

    useFlowStore.getState().undo();
    expect(useFlowStore.getState().graph.nodes[0].label).toBeUndefined();
  });

  it("undo after node delete restores the node", () => {
    const s = useFlowStore.getState();
    s.addNode("dataset", { x: 0, y: 0 });
    useFlowStore.setState({ past: [], future: [] });

    // Simulate a remove change
    useFlowStore.getState().applyNodesChanges([{ id: "n1", type: "remove" }]);
    expect(useFlowStore.getState().graph.nodes).toHaveLength(0);
    expect(useFlowStore.getState().past).toHaveLength(1);

    useFlowStore.getState().undo();
    expect(useFlowStore.getState().graph.nodes).toHaveLength(1);
    expect(useFlowStore.getState().graph.nodes[0].kind).toBe("dataset");
  });

  it("undo after edge delete restores the edge", () => {
    const s = useFlowStore.getState();
    s.addNode("dataset", { x: 0, y: 0 });
    s.addNode("prepare", { x: 100, y: 0 });
    useFlowStore.getState().connect({
      source: "n1", target: "n2", sourceHandle: "cleaned", targetHandle: "raw-dir",
    });
    useFlowStore.setState({ past: [], future: [] });

    // Simulate an edge remove
    useFlowStore.getState().applyEdgesChanges([{ id: "e1", type: "remove" }]);
    expect(useFlowStore.getState().graph.edges).toHaveLength(0);
    expect(useFlowStore.getState().past).toHaveLength(1);

    useFlowStore.getState().undo();
    expect(useFlowStore.getState().graph.edges).toHaveLength(1);
  });

  it("position-only changes do NOT push history", () => {
    const s = useFlowStore.getState();
    s.addNode("dataset", { x: 0, y: 0 });
    useFlowStore.setState({ past: [], future: [] });

    // Position change should not push history
    useFlowStore.getState().applyNodesChanges([
      { id: "n1", type: "position", position: { x: 50, y: 50 } },
    ]);
    expect(useFlowStore.getState().past).toHaveLength(0);
  });

  it("clear() resets both stacks", () => {
    const s = useFlowStore.getState();
    s.addNode("dataset", { x: 0, y: 0 });
    s.undo();
    expect(useFlowStore.getState().past).toHaveLength(0);
    expect(useFlowStore.getState().future).toHaveLength(1);

    useFlowStore.getState().clear();
    expect(useFlowStore.getState().past).toHaveLength(0);
    expect(useFlowStore.getState().future).toHaveLength(0);
  });
});