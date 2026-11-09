import { describe, expect, it } from "vitest";
import { walkModelGraph, type WalkError } from "./graphWalk";
import { LAYER_REGISTRY } from "./registry";
import { pTotalDense } from "./paramMath";

const R = LAYER_REGISTRY;

function node(id: string, kind: string, props: Record<string, unknown> = {}) {
  return { id, kind, props, position: { x: 0, y: 0 } };
}
function edge(id: string, from: string, to: string, fromPort: string, toPort: string) {
  return { id, from, to, fromPort, toPort };
}
function errText(errors: WalkError[]): string {
  return errors.map((e) => e.nodeId + ": " + e.message).join("\n");
}

describe("walkModelGraph — linear body chain (a)", () => {
  // input -> embedding -> rmsnorm -> rope -> swiglu -> lmHead (untied)
  const graph = {
    nodes: [
      node("t", "input", { vocab_size: 100, ctx: 8 }),
      node("emb", "embedding", { vocab_size: 100, d: 16 }),
      node("n1", "rmsnorm"),
      node("rp", "rope"),
      node("ffn", "swigluFfn", { ffn: 32 }),
      node("head", "lmHead", { vocab_size: 100, tied: false }),
    ],
    edges: [
      edge("e1", "t", "emb", "tokens", "tokens"),
      edge("e2", "emb", "n1", "out", "x"),
      edge("e3", "n1", "rp", "out", "x"),
      edge("e4", "rp", "ffn", "out", "x"),
      edge("e5", "ffn", "head", "out", "hidden"),
    ],
  };

  it("infers correct shapes along the chain", () => {
    const r = walkModelGraph(graph, R);
    expect(r.errors).toEqual([]);
    expect(r.shapes.get("t")).toEqual({ s: 8, d: 100 });
    expect(r.shapes.get("emb")).toEqual({ s: 8, d: 16 });
    expect(r.shapes.get("n1")).toEqual({ s: 8, d: 16 });
    expect(r.shapes.get("rp")).toEqual({ s: 8, d: 16 });
    expect(r.shapes.get("ffn")).toEqual({ s: 8, d: 16 });
    expect(r.shapes.get("head")).toEqual({ s: 8, d: 100 });
  });

  it("rolls up params and total exactly", () => {
    const r = walkModelGraph(graph, R);
    expect(r.params).toEqual({
      t: 0,
      emb: 1600, // 100 * 16
      n1: 16, // d
      rp: 0,
      ffn: 1536, // 3 * 16 * 32
      head: 1600, // untied: 100 * 16
    });
    expect(r.total).toBe(4752);
  });
});

describe("walkModelGraph — cycle detection (b)", () => {
  it("reports 'cycle detected' per cycle node, no hang, other nodes unaffected", () => {
    const graph = {
      nodes: [
        node("t", "input", { vocab_size: 64, ctx: 4 }),
        node("a", "rmsnorm"),
        node("b", "rope"),
      ],
      edges: [
        edge("e1", "a", "b", "out", "x"),
        edge("e2", "b", "a", "out", "x"),
      ],
    };
    const r = walkModelGraph(graph, R);
    const msgs = r.errors.map((e) => e.message);
    expect(msgs.some((m) => /cycle detected/i.test(m))).toBe(true);
    // BOTH cycle nodes carry the error
    const cycleIds = r.errors.filter((e) => /cycle detected/i.test(e.message)).map((e) => e.nodeId);
    expect(new Set(cycleIds)).toEqual(new Set(["a", "b"]));
    // cycle nodes produce no shape
    expect(r.shapes.has("a")).toBe(false);
    expect(r.shapes.has("b")).toBe(false);
    // independent component still walks
    expect(r.shapes.get("t")).toEqual({ s: 4, d: 64 });
    expect(r.total).toBe(0);
  });
});

describe("walkModelGraph — missing inputs (c)", () => {
  it("gqaAttention with only q wired errors per missing port by name", () => {
    const graph = {
      nodes: [
        node("t", "input", { vocab_size: 64, ctx: 4 }),
        node("emb", "embedding", { vocab_size: 64, d: 32 }),
        node("attn", "gqaAttention", { heads: 4, kv_heads: 2 }),
      ],
      edges: [
        edge("e1", "t", "emb", "tokens", "tokens"),
        edge("e2", "emb", "attn", "out", "q"),
      ],
    };
    const r = walkModelGraph(graph, R);
    const attnErrs = r.errors.filter((e) => e.nodeId === "attn").map((e) => e.message);
    expect(attnErrs.some((m) => m.includes("missing input k"))).toBe(true);
    expect(attnErrs.some((m) => m.includes("missing input v"))).toBe(true);
    expect(r.shapes.has("attn")).toBe(false);
    expect(r.params["attn"]).toBeUndefined();
    // upstream nodes are untouched
    expect(r.shapes.get("emb")).toEqual({ s: 4, d: 32 });
  });
});

describe("walkModelGraph — smoke-anchor mini graph (d)", () => {
  // Locked composition: input(ctx 256, vocab 32768) -> embedding(d 256);
  // FOUR explicit per-layer blocks [rope -> gqaAttention(q,k,v) -> swigluFfn];
  // one layerStack(depth 4, d 256) carries ONLY the outer norms (2N+1)*d.
  // Body per layer = 196608 + 786432 = 983040; total must tie pTotalDense.
  const blocks = [1, 2, 3, 4].flatMap((i) => [
    node("rope" + i, "rope"),
    node("attn" + i, "gqaAttention", { heads: 4, kv_heads: 2 }),
    node("ffn" + i, "swigluFfn", { ffn: 1024 }),
  ]);
  const graph = {
    nodes: [
      node("t", "input", { vocab_size: 32768, ctx: 256 }),
      node("emb", "embedding", { vocab_size: 32768, d: 256 }),
      node("stack", "layerStack", { depth: 4, d: 256, heads: 4, kv: 2, ffn: 1024 }),
      ...blocks,
    ],
    edges: [
      edge("w1", "t", "emb", "tokens", "tokens"),
      edge("w2", "emb", "stack", "out", "x"),
      ...[1, 2, 3, 4].flatMap((i) => [
        edge("wr" + i, "emb", "rope" + i, "out", "x"),
        edge("wq" + i, "rope" + i, "attn" + i, "out", "q"),
        edge("wk" + i, "rope" + i, "attn" + i, "out", "k"),
        edge("wv" + i, "rope" + i, "attn" + i, "out", "v"),
        edge("wf" + i, "attn" + i, "ffn" + i, "out", "x"),
      ]),
    ],
  };

  it("per-layer params: gqa 196608 + swiglu 786432 = 983040 (rope 0)", () => {
    const r = walkModelGraph(graph, R);
    expect(r.errors).toEqual([]);
    for (const i of [1, 2, 3, 4]) {
      expect(r.params["rope" + i]).toBe(0);
      expect(r.params["attn" + i]).toBe(196_608);
      expect(r.params["ffn" + i]).toBe(786_432);
    }
    const perLayer = r.params["attn1"] + r.params["ffn1"] + r.params["rope1"];
    expect(perLayer).toBe(983_040);
    // layerStack counts ONLY outer norms: (2*4+1)*256
    expect(r.params["stack"]).toBe(2_304);
    expect(r.params["emb"]).toBe(8_388_608);
  });

  it("total ties pTotalDense = 12,323,072 exactly", () => {
    const r = walkModelGraph(graph, R);
    const expected = 8_388_608 + 4 * 983_040 + 2_304;
    expect(r.total).toBe(expected);
    expect(r.total).toBe(
      pTotalDense({ layers: 4, d: 256, heads: 4, kv: 2, ffn: 1024, vocab: 32_768 }),
    );
    expect(r.total).toBe(12_323_072);
  });

  it("gqaAttention shape propagates {s:256, d:256} through the body", () => {
    const r = walkModelGraph(graph, R);
    expect(r.shapes.get("t")).toEqual({ s: 256, d: 32768 });
    expect(r.shapes.get("emb")).toEqual({ s: 256, d: 256 });
    expect(r.shapes.get("attn2")).toEqual({ s: 256, d: 256 });
    expect(r.shapes.get("stack")).toEqual({ s: 256, d: 256 });
  });
});

describe("walkModelGraph — residual d mismatch (e)", () => {
  it("surfacing the registry error with both d values", () => {
    const graph = {
      nodes: [
        node("t", "input", { vocab_size: 64, ctx: 4 }),
        node("e256", "embedding", { vocab_size: 64, d: 256 }),
        node("e512", "embedding", { vocab_size: 64, d: 512 }),
        node("res", "residual"),
      ],
      edges: [
        edge("e1", "t", "e256", "tokens", "tokens"),
        edge("e2", "t", "e512", "tokens", "tokens"),
        edge("e3", "e256", "res", "out", "stream"),
        edge("e4", "e512", "res", "out", "bypass"),
      ],
    };
    const r = walkModelGraph(graph, R);
    expect(errText(r.errors)).toMatch(/residual: bypass d=512 must equal stream d=256/);
    expect(r.errors.some((e) => e.nodeId === "res")).toBe(true);
    expect(r.shapes.has("res")).toBe(false);
    // both embedding branches still walked
    expect(r.shapes.get("e256")).toEqual({ s: 4, d: 256 });
    expect(r.shapes.get("e512")).toEqual({ s: 4, d: 512 });
  });
});

describe("walkModelGraph — port-name honesty + duplicate wiring", () => {
  it("an edge naming a nonexistent input port is an honest error and leaves the port missing", () => {
    const graph = {
      nodes: [
        node("t", "input", { vocab_size: 64, ctx: 4 }),
        node("emb", "embedding", { vocab_size: 64, d: 32 }),
      ],
      edges: [edge("e1", "t", "emb", "tokens", "WRONG_PORT")],
    };
    const r = walkModelGraph(graph, R);
    expect(errText(r.errors)).toMatch(/WRONG_PORT/);
    expect(errText(r.errors)).toMatch(/missing input tokens/);
    expect(r.shapes.has("emb")).toBe(false);
  });

  it("an edge naming a nonexistent output port errors on the source node", () => {
    const graph = {
      nodes: [
        node("t", "input", { vocab_size: 64, ctx: 4 }),
        node("emb", "embedding", { vocab_size: 64, d: 32 }),
      ],
      edges: [edge("e1", "t", "emb", "NOPE", "tokens")],
    };
    const r = walkModelGraph(graph, R);
    expect(r.errors.some((e) => e.nodeId === "t" && /NOPE/.test(e.message))).toBe(true);
    // single-output spec: the shape still flows; source params excluded from total
    expect(r.shapes.get("emb")).toEqual({ s: 4, d: 32 });
    // total excludes only the ERRORED node's params (t: 0 anyway); emb did
    // not error, so its params stay counted: 64*32 = 2048
    expect(r.total).toBe(2048);
  });

  it("two edges into the SAME input port are an error; node produces no shape", () => {
    const graph = {
      nodes: [
        node("t", "input", { vocab_size: 64, ctx: 4 }),
        node("e1n", "embedding", { vocab_size: 64, d: 32 }),
        node("e2n", "embedding", { vocab_size: 64, d: 48 }),
        node("n1", "rmsnorm"),
      ],
      edges: [
        edge("ea", "e1n", "n1", "out", "x"),
        edge("eb", "e2n", "n1", "out", "x"),
      ],
    };
    const r = walkModelGraph(graph, R);
    expect(errText(r.errors)).toMatch(/more than one incoming edge/);
    expect(r.shapes.has("n1")).toBe(false);
  });

  it("unknown kind errors on that node without aborting the walk", () => {
    const graph = {
      nodes: [
        node("t", "input", { vocab_size: 64, ctx: 4 }),
        node("w", "warpDrive", {}),
      ],
      edges: [],
    };
    const r = walkModelGraph(graph, R);
    expect(errText(r.errors)).toMatch(/warpDrive/);
    expect(errText(r.errors)).toMatch(/unknown kind/i);
    expect(r.shapes.get("t")).toEqual({ s: 4, d: 64 });
  });

  it("empty graph walks to an empty result", () => {
    const r = walkModelGraph({ nodes: [], edges: [] }, R);
    expect(r.errors).toEqual([]);
    expect(r.total).toBe(0);
    expect(r.shapes.size).toBe(0);
  });
});
