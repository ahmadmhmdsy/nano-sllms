/* ------------------------------------------------------------------ */
/* Task M2.2: node search unit tests. The required cases - filter by   */
/* kind, filter by label, Enter cycling - target the pure functions    */
/* in ./stores/search directly; the store-wiring and toXYNodes         */
/* className cases pin the integration without a DOM.                  */
/* ------------------------------------------------------------------ */
import { describe, it, expect, beforeEach } from "vitest";
import { searchMatches, nextMatchIndex } from "./stores/search";
import { useFlowStore, toXYNodes } from "./store";
import type { Graph, RegistrySnapshot } from "./store";

const REGISTRY: RegistrySnapshot = {
  valid_kinds: ["dataset", "prepare", "train"],
  nodes: {
    dataset: { ports: [], props: [], numeric_props: [], gate: null, label_semantic: null, features: [] },
    prepare: { ports: [], props: [], numeric_props: [], gate: null, label_semantic: null, features: [] },
    train: { ports: [], props: [], numeric_props: [], gate: null, label_semantic: null, features: [] },
  },
  gates: { dataset: null, prepare: null, train: null },
};

function node(id: string, kind: string, x: number, label?: string) {
  return { id, kind, ...(label !== undefined ? { label } : {}), props: {}, position: { x, y: 0 } };
}

/* Two prepare nodes on purpose: a kind query must return BOTH, in
 * graph order (that order is what Enter cycles through). */
const graph: Graph = {
  nodes: [
    node("n1", "dataset", 0, "RowTour-360"),
    node("n2", "prepare", 100, "clean rows"),
    node("n3", "prepare", 200, "dedupe"),
    node("n4", "train", 300, "nano"),
  ],
  edges: [],
};

describe("searchMatches - filter by kind", () => {
  it("returns every node whose kind contains the query (case-insensitive)", () => {
    expect(searchMatches(graph, "prepare")).toEqual(["n2", "n3"]);
    expect(searchMatches(graph, "PREPARE")).toEqual(["n2", "n3"]);
    expect(searchMatches(graph, "train")).toEqual(["n4"]);
    expect(searchMatches(graph, "dataset")).toEqual(["n1"]);
  });

  it("a label match does not pull in same-kind nodes (no leak across fields)", () => {
    // "dedupe" is n3's label; n2 shares the kind but must not match.
    expect(searchMatches(graph, "dedupe")).toEqual(["n3"]);
    // "nano" is n4's label; the query matches by label only.
    expect(searchMatches(graph, "nano")).toEqual(["n4"]);
  });
});

describe("searchMatches - filter by label", () => {
  it("returns nodes whose label contains the query (case-insensitive)", () => {
    expect(searchMatches(graph, "rowtour")).toEqual(["n1"]);
    expect(searchMatches(graph, "RowTour-360")).toEqual(["n1"]);
    expect(searchMatches(graph, "dedupe")).toEqual(["n3"]);
  });

  it("label substring matches work mid-string", () => {
    expect(searchMatches(graph, "clean")).toEqual(["n2"]);
  });
});

describe("searchMatches - filter by id and query hygiene", () => {
  it("matches node ids", () => {
    expect(searchMatches(graph, "n1")).toEqual(["n1"]);
    expect(searchMatches(graph, "n")).toEqual(["n1", "n2", "n3", "n4"]);
  });

  it("an empty or whitespace query matches nothing (no-search state)", () => {
    expect(searchMatches(graph, "")).toEqual([]);
    expect(searchMatches(graph, "   ")).toEqual([]);
  });

  it("a query matching nothing returns []", () => {
    expect(searchMatches(graph, "warp")).toEqual([]);
  });

  it("nodes without a label simply do not match on label", () => {
    const noLabel: Graph = { nodes: [node("n9", "tokenize", 0)], edges: [] };
    expect(searchMatches(noLabel, "tokenize")).toEqual(["n9"]);
    expect(searchMatches(noLabel, "")).toEqual([]);
  });
});

describe("nextMatchIndex - Enter cycling", () => {
  const matches = ["n2", "n3", "n4"];

  it("starts at the first match when nothing is selected", () => {
    expect(nextMatchIndex(matches, null)).toBe(0);
  });

  it("advances to the next match after the current selection", () => {
    expect(nextMatchIndex(matches, "n2")).toBe(1);
    expect(nextMatchIndex(matches, "n3")).toBe(2);
  });

  it("wraps around to the first match after the last", () => {
    expect(nextMatchIndex(matches, "n4")).toBe(0);
  });

  it("a selection outside the match list restarts at the first match", () => {
    expect(nextMatchIndex(matches, "n1")).toBe(0);
    expect(nextMatchIndex(matches, null)).toBe(0);
  });

  it("empty matches yield index 0 (callers no-op on it)", () => {
    expect(nextMatchIndex([], null)).toBe(0);
    expect(nextMatchIndex([], "n2")).toBe(0);
  });
});

/* ------------------------------------------------------------------ */
/* Store wiring: the ui slice owns the query; Enter cycles through     */
/* matches via cycleSearchMatch (selection + centerRequest).           */
/* ------------------------------------------------------------------ */

describe("node search store wiring (M2.2)", () => {
  beforeEach(() => {
    useFlowStore.setState({
      graph,
      registry: REGISTRY,
      searchQuery: "",
      centerRequest: null,
      selection: null,
      error: null,
    });
  });

  it("setSearchQuery updates the query; clear restores no-search", () => {
    useFlowStore.getState().setSearchQuery("prepare");
    expect(useFlowStore.getState().searchQuery).toBe("prepare");
    useFlowStore.getState().setSearchQuery("");
    expect(useFlowStore.getState().searchQuery).toBe("");
  });

  it("Enter selects the first match and issues a center request at its position", () => {
    useFlowStore.getState().setSearchQuery("prepare");
    useFlowStore.getState().cycleSearchMatch();
    const st = useFlowStore.getState();
    expect(st.selection).toBe("n2");
    expect(st.centerRequest).not.toBeNull();
    expect(st.centerRequest?.nodeId).toBe("n2");
    expect(st.centerRequest?.x).toBe(100);
    expect(st.centerRequest?.y).toBe(0);
  });

  it("a second Enter cycles to the next match", () => {
    useFlowStore.getState().setSearchQuery("prepare");
    useFlowStore.getState().cycleSearchMatch();
    useFlowStore.getState().cycleSearchMatch();
    const st = useFlowStore.getState();
    expect(st.selection).toBe("n3");
    expect(st.centerRequest?.nodeId).toBe("n3");
    expect(st.centerRequest?.x).toBe(200);
  });

  it("cycling wraps from the last match back to the first", () => {
    useFlowStore.getState().setSearchQuery("prepare");
    useFlowStore.getState().cycleSearchMatch(); // n2
    useFlowStore.getState().cycleSearchMatch(); // n3
    useFlowStore.getState().cycleSearchMatch(); // wrap -> n2
    expect(useFlowStore.getState().selection).toBe("n2");
  });

  it("a selection outside the matches restarts at the first match", () => {
    useFlowStore.setState({ selection: "n4" });
    useFlowStore.getState().setSearchQuery("prepare");
    useFlowStore.getState().cycleSearchMatch();
    expect(useFlowStore.getState().selection).toBe("n2");
  });

  it("Enter with no matches is a no-op (selection and centerRequest untouched)", () => {
    useFlowStore.setState({ selection: "n1" });
    useFlowStore.getState().setSearchQuery("warp");
    useFlowStore.getState().cycleSearchMatch();
    const st = useFlowStore.getState();
    expect(st.selection).toBe("n1");
    expect(st.centerRequest).toBeNull();
  });

  it("repeated requests get fresh nonces so FlowCanvas answers each one", () => {
    useFlowStore.getState().setSearchQuery("n"); // matches n1..n4
    useFlowStore.getState().cycleSearchMatch();
    const first = useFlowStore.getState().centerRequest?.nonce;
    useFlowStore.getState().cycleSearchMatch();
    const second = useFlowStore.getState().centerRequest?.nonce;
    expect(first).toBeDefined();
    expect(second).toBeDefined();
    expect(second).not.toBe(first);
  });
});

/* ------------------------------------------------------------------ */
/* Canvas projection: toXYNodes maps the query onto per-node           */
/* classNames (node-highlight / node-dim) when a search is active.     */
/* ------------------------------------------------------------------ */

describe("toXYNodes search classNames (canvas highlight wiring)", () => {
  it("matching nodes get node-highlight, non-matching get node-dim", () => {
    const xy = toXYNodes(graph, REGISTRY, [], "idle", null, "prepare");
    expect(xy.map((n) => n.className)).toEqual([
      "node-dim",
      "node-highlight",
      "node-highlight",
      "node-dim",
    ]);
  });

  it("an empty query leaves every className undefined (no highlight state)", () => {
    const xy = toXYNodes(graph, REGISTRY);
    expect(xy.every((n) => n.className === undefined)).toBe(true);
    const xyBlank = toXYNodes(graph, REGISTRY, [], "idle", null, "   ");
    expect(xyBlank.every((n) => n.className === undefined)).toBe(true);
  });

  it("className does not disturb the runState data identity path", () => {
    const idle = toXYNodes(graph, REGISTRY);
    const searched = toXYNodes(graph, REGISTRY, idle, "idle", null, "train");
    expect(searched[3].className).toBe("node-highlight");
    expect(searched[3].data).toBe(idle[3].data); // unchanged domain node
    expect(searched[0].className).toBe("node-dim");
  });
});
