import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import {
  WEBUI_URL,
  INFER_CHECKPOINT_NOTE,
  inferHandoffDialog,
  attemptWebuiOpen,
} from "./inferHandoff";
import { useFlowStore, validateConnect, connectReducer, toXYNodes, type RegistrySnapshot } from "./store";

/* Task 11 — infer render-only shortcut node: pure pieces + click path. */

const REGISTRY: RegistrySnapshot = {
  valid_kinds: ["dataset", "prepare", "tokenize", "train", "eval", "infer"],
  nodes: {
    ...Object.fromEntries([]),
  } as RegistrySnapshot["nodes"],
  gates: {},
};
// Minimal slice for the connect test (train -> infer ckpt-dir edge).
REGISTRY.nodes = {
  train: {
    ports: [
      { name: "shard-dir", direction: "in", type: "shard-dir" },
      { name: "ckpt-dir", direction: "out", type: "ckpt-dir" },
    ],
    props: [],
    numeric_props: [],
    gate: "gpu",
  },
  infer: {
    ports: [{ name: "ckpt-dir", direction: "in", type: "ckpt-dir" }],
    props: [],
    numeric_props: [],
    gate: "gpu/cpu",
  },
} as RegistrySnapshot["nodes"];
REGISTRY.gates = { train: "gpu", infer: "gpu/cpu" };

describe("Task 11 infer shortcut pure pieces", () => {
  it("WEBUI_URL is the webui/app.py launch default (CHAT_PORT 7860)", () => {
    expect(WEBUI_URL).toBe("http://127.0.0.1:7860");
  });

  it("checkpoint note is the honest no-fabricated-path copy", () => {
    expect(INFER_CHECKPOINT_NOTE).toBe("checkpoint: see connected input node's output");
    expect(INFER_CHECKPOINT_NOTE).not.toMatch(/runs\//);
  });

  it("dialog info carries the manual venv launch instructions", () => {
    const info = inferHandoffDialog();
    expect(info.launchCommand).toContain("webui/app.py");
    expect(info.launchCommand).toContain(".venv/Scripts/python.exe");
    expect(info.checkpointNote).toBe(INFER_CHECKPOINT_NOTE);
  });

  /* The kind-string render checker is gone: the button visibility now
   * comes from the registry spec's features flag (spec.features.includes(
   * CHAT_BUTTON_FEATURE)), pinned in store.test.ts T3 wiring tests. */
  it("toXYNodes keeps the infer node's single in-port (wire-consistent F5)", () => {
    const xy = toXYNodes(
      { nodes: [{ id: "n1", kind: "infer", props: {}, position: { x: 0, y: 0 } }], edges: [] },
      REGISTRY,
    );
    const ports = xy[0].data.ports.filter((p) => p.direction === "in");
    expect(ports).toHaveLength(1);
    expect(ports[0].type).toBe("ckpt-dir");
  });

  it("train -> infer ckpt-dir edge stays valid (connect reducer)", () => {
    const graph = {
      nodes: [
        { id: "n1", kind: "train", props: {}, position: { x: 0, y: 0 } },
        { id: "n2", kind: "infer", props: {}, position: { x: 1, y: 0 } },
      ],
      edges: [],
    };
    const res = connectReducer(REGISTRY, graph, {
      source: "n1", target: "n2", sourceHandle: "ckpt-dir", targetHandle: "ckpt-dir",
    });
    expect(res.error).toBeNull();
    expect(validateConnect(REGISTRY, res.graph, "n1", "n2", "ckpt-dir", "ckpt-dir")).toEqual([]);
  });
});

describe("Task 11 handoff click path (store + window.open spy)", () => {
  /* node environment: no real window -> stub the global the click path
   * uses; the spy then asserts the attempt exactly as in a browser. */
  let fakeWindow: { open: ReturnType<typeof vi.fn> };
  beforeEach(() => {
    fakeWindow = { open: vi.fn(() => null) };
    vi.stubGlobal("window", fakeWindow);
    useFlowStore.setState({ inferHandoffNodeId: null, error: null });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("openInferHandoff sets the dialog node id and the click attempts window.open(WEBUI_URL)", () => {
    const spy = vi.spyOn(window, "open"); // stub returns null = popup blocked
    // The click path: openInferHandoff(nodeId) + attemptWebuiOpen() — the
    // dialog shows MANUAL instructions even though open() returned null.
    useFlowStore.getState().openInferHandoff("n1");
    const attempt = attemptWebuiOpen();
    expect(spy).toHaveBeenCalledOnce();
    expect(spy).toHaveBeenCalledWith(WEBUI_URL, "_blank");
    expect(attempt).toBeNull();
    expect(useFlowStore.getState().inferHandoffNodeId).toBe("n1");
    useFlowStore.getState().dismissInferHandoff();
    expect(useFlowStore.getState().inferHandoffNodeId).toBeNull();
  });

  it("attemptWebuiOpen forwards a successful open in the same shape", () => {
    const fakeWin = { focus: () => {} } as unknown as Window;
    fakeWindow.open.mockImplementation(() => fakeWin);
    const spy = vi.spyOn(window, "open");
    expect(attemptWebuiOpen()).toBe(fakeWin);
    expect(spy).toHaveBeenCalledWith(WEBUI_URL, "_blank");
  });
});
