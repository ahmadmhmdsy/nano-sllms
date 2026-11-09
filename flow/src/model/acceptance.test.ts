/**
 * F2 plan Task 6 (acceptance hard gate): the committed example graphs under
 * flow/models/ must walk through the REAL editor engine and produce the
 * repo decoders' param totals EXACTLY (dense GQA formula anchors — numbers
 * are pinned twice: against the engine-internal pTotalDense AND the raw
 * integer). Wiring errors surface as walk errors here, never silently.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { walkModelGraph } from "./graphWalk";
import { LAYER_REGISTRY } from "./registry";
import { pTotalDense } from "./paramMath";

interface RawGraph {
  format: string;
  name: string;
  nodes: { id: string; kind: string; props: Record<string, unknown>; position: { x: number; y: number } }[];
  edges: { id: string; from: string; to: string; fromPort: string; toPort: string }[];
}

function load(name: string): RawGraph {
  // vitest cwd is flow/, so the committed flow/models dir is models/ here.
  return JSON.parse(readFileSync(join("models", name + ".modelgraph.json"), "utf8")) as RawGraph;
}

describe("acceptance: committed decoder graphs walk to exact totals", () => {
  const cases = [
    { name: "smoke", total: 12_323_072, dim: { layers: 4, d: 256, heads: 4, kv: 2, ffn: 1024, vocab: 32_768 } },
    { name: "pilot", total: 100_682_496, dim: { layers: 12, d: 768, heads: 12, kv: 4, ffn: 2048, vocab: 32_768 } },
    { name: "target", total: 226_526_208, dim: { layers: 16, d: 1024, heads: 16, kv: 4, ffn: 3072, vocab: 32_768 } },
  ];

  for (const g of cases) {
    it(g.name + " totals " + g.total.toLocaleString() + " — wiring honest, ties pTotalDense", () => {
      const doc = load(g.name);
      expect(doc.format).toBe("model/0.1");
      const walked = walkModelGraph(doc, LAYER_REGISTRY);
      expect(walked.errors).toEqual([]);
      expect(walked.total).toBe(g.total);
      expect(walked.total).toBe(pTotalDense(g.dim));
      // honest topology spot: layerStack carries exactly the OUTER norms
      const stack = doc.nodes.find((n) => n.kind === "layerStack");
      expect(stack).toBeDefined();
      expect(walked.params[stack!.id]).toBe((2 * g.dim.layers + 1) * g.dim.d);
      // tied embeddings: lmHead costs 0 params
      const head = doc.nodes.find((n) => n.kind === "lmHead");
      expect(head!.props.tied).toBe(true);
      expect(walked.params[head!.id]).toBe(0);
    });
  }
});
