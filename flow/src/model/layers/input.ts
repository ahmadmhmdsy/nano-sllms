import type { LayerSpec, PropSpec } from "./shared";
import { numProp } from "./shared";

const VOCAB_SIZE: PropSpec = { name: "vocab_size", kind: "int", default: 32768, min: 1 };
const CTX: PropSpec = { name: "ctx", kind: "int", default: 1024, min: 1 };

/** input: token-id source. 0 inputs; d carries the vocab index. */
export const input: LayerSpec = {
  kind: "input",
  label: "Input Tokens",
  summary: "Token-id stream; d carries the vocab index, s is sequence length.",
  inputs: [],
  outputs: [{ name: "tokens", label: "Token IDs" }],
  props: [VOCAB_SIZE, CTX],
  inferShape(props, _inputs) {
    const vocabSize = numProp("vocab_size", props.vocab_size, VOCAB_SIZE);
    const ctx = numProp("ctx", props.ctx, CTX);
    return { s: ctx, d: vocabSize };
  },
  paramCount() {
    return 0;
  },
};
