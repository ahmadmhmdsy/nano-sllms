import type { LayerSpec, TensorPort } from "./shared";
import { requireIn } from "./shared";

const IN: TensorPort = { name: "x", label: "Hidden" };

/** rmsnorm: d passthrough; weight-only, params = d. */
export const rmsnorm: LayerSpec = {
  kind: "rmsnorm",
  label: "RMSNorm",
  summary: "RMSNorm over the last dim; learnable weight only (d params).",
  inputs: [IN],
  outputs: [{ name: "out", label: "Hidden" }],
  props: [],
  inferShape(_props, inputs) {
    return { ...requireIn(inputs, 0, "rmsnorm", IN) };
  },
  paramCount(_props, out) {
    if (typeof out.d !== "number" || !(out.d > 0)) {
      throw new Error("rmsnorm: output shape has no positive d");
    }
    return out.d;
  },
};
