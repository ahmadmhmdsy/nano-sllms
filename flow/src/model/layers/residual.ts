import type { LayerSpec, TensorPort } from "./shared";
import { requireIn } from "./shared";

const STREAM: TensorPort = { name: "stream", label: "Stream" };
const BYPASS: TensorPort = { name: "bypass", label: "Bypass" };

/** residual: add two streams; bypass d must equal stream d. */
export const residual: LayerSpec = {
  kind: "residual",
  label: "Residual Add",
  summary: "Element-wise add of the main stream with the bypass path; same d required.",
  inputs: [STREAM, BYPASS],
  outputs: [{ name: "out", label: "Hidden" }],
  props: [],
  inferShape(_props, inputs) {
    const stream = requireIn(inputs, 0, "residual", STREAM);
    const bypass = requireIn(inputs, 1, "residual", BYPASS);
    if (bypass.d !== stream.d) {
      throw new Error(`residual: bypass d=${bypass.d} must equal stream d=${stream.d}`);
    }
    return { ...stream };
  },
  paramCount() {
    return 0;
  },
};
