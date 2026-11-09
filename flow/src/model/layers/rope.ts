import type { LayerSpec, TensorPort } from "./shared";
import { requireIn } from "./shared";

const IN: TensorPort = { name: "x", label: "Hidden" };

/** rope: rotary positional embedding; passthrough, no params. */
export const rope: LayerSpec = {
  kind: "rope",
  label: "RoPE",
  summary: "Rotary positional embedding; passthrough {s,d}, no parameters.",
  inputs: [IN],
  outputs: [{ name: "out", label: "Hidden" }],
  props: [],
  inferShape(_props, inputs) {
    return { ...requireIn(inputs, 0, "rope", IN) };
  },
  paramCount() {
    return 0;
  },
};
