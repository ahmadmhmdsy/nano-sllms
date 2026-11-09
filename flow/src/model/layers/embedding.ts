import type { LayerSpec, PropSpec, TensorPort } from "./shared";
import { numProp, requireIn } from "./shared";

import { pEmbed } from "../paramMath";

const TOKENS: TensorPort = { name: "tokens", label: "Token IDs" };

const VOCAB_SIZE: PropSpec = { name: "vocab_size", kind: "int", default: 32768, min: 1 };
const D: PropSpec = { name: "d", kind: "int", default: 256, min: 1 };

/** embedding: vocab*d weight. Propagate s from input. */
export const embedding: LayerSpec = {
  kind: "embedding",
  label: "Embedding",
  summary: "Embedding table lookup (vocab_size x d); propagates sequence length.",
  inputs: [TOKENS],
  outputs: [{ name: "out", label: "Hidden" }],
  props: [VOCAB_SIZE, D],
  inferShape(props, inputs) {
    const inSh = requireIn(inputs, 0, "embedding", TOKENS);
    const d = numProp("d", props.d, D);
    return { ...inSh, d };
  },
  paramCount(props) {
    return pEmbed({
      vocab: numProp("vocab_size", props.vocab_size, VOCAB_SIZE),
      d: numProp("d", props.d, D),
    });
  },
};
