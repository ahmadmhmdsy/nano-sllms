import type { LayerSpec, PropSpec, TensorPort } from "./shared";
import { numProp, requireIn } from "./shared";

const Q: TensorPort = { name: "q", label: "Query" };
const K: TensorPort = { name: "k", label: "Key" };
const V: TensorPort = { name: "v", label: "Value" };
import { pAttentionDense } from "../paramMath";

const HEADS: PropSpec = { name: "heads", kind: "int", default: 4, min: 1 };
const KV_HEADS: PropSpec = { name: "kv_heads", kind: "int", default: 2, min: 1 };

/** gqaAttention: THREE inputs (q, k, v) — LOCKED wiring; all must share d. */
export const gqaAttention: LayerSpec = {
  kind: "gqaAttention",
  label: "GQA Attention",
  summary: "Grouped-query attention (dense): q/k/v inputs; out d = q d.",
  inputs: [Q, K, V],
  outputs: [{ name: "out", label: "Hidden" }],
  props: [HEADS, KV_HEADS],
  inferShape(props, inputs) {
    const q = requireIn(inputs, 0, "gqaAttention", Q);
    requireIn(inputs, 1, "gqaAttention", K);
    requireIn(inputs, 2, "gqaAttention", V);
    for (const [port, s] of [["k", inputs[1]], ["v", inputs[2]]] as const) {
      if (s.d !== q.d) {
        throw new Error(`gqaAttention: input '${port}' has d=${s.d} but q has d=${q.d} — all three inputs must carry the same d`);
      }
    }
    const heads = numProp("heads", props.heads, HEADS);
    const kvHeads = numProp("kv_heads", props.kv_heads, KV_HEADS);
    if (kvHeads > heads) {
      throw new Error(`gqaAttention: kv_heads (${kvHeads}) must be <= heads (${heads})`);
    }
    if (heads > q.d) {
      throw new Error(`gqaAttention: heads (${heads}) must be <= d (${q.d})`);
    }
    return { ...q };
  },
  paramCount(props, out) {
    if (typeof out.d !== "number" || !(out.d > 0)) {
      throw new Error("gqaAttention: output shape has no positive d");
    }
    return pAttentionDense({
      d: out.d,
      heads: numProp("heads", props.heads, HEADS),
      kv: numProp("kv_heads", props.kv_heads, KV_HEADS),
    });
  },
};
