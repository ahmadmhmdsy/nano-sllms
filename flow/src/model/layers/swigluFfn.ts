import type { LayerSpec, PropSpec, TensorPort } from "./shared";
import { numProp, requireIn } from "./shared";

const X: TensorPort = { name: "x", label: "Hidden" };
import { pFFNDense } from "../paramMath";

const FFN: PropSpec = { name: "ffn", kind: "int", default: 1024, min: 1 };

/** swigluFfn: gate+up+down (SwiGLU), params = 3*d*ffn. */
export const swigluFfn: LayerSpec = {
  kind: "swigluFfn",
  label: "SwiGLU FFN",
  summary: "SwiGLU feed-forward block (gate+up+down); d passthrough.",
  inputs: [X],
  outputs: [{ name: "out", label: "Hidden" }],
  props: [FFN],
  inferShape(_props, inputs) {
    return { ...requireIn(inputs, 0, "swigluFfn", X) };
  },
  paramCount(props, out) {
    if (typeof out.d !== "number" || !(out.d > 0)) {
      throw new Error("swigluFfn: output shape has no positive d");
    }
    return pFFNDense({
      d: out.d,
      ffn: numProp("ffn", props.ffn, FFN),
    });
  },
};
