import type { LayerSpec, PropSpec, TensorPort } from "./shared";
import { boolProp, numProp, requireIn } from "./shared";

const HIDDEN: TensorPort = { name: "hidden", label: "Hidden" };
import { pEmbed } from "../paramMath";

const VOCAB_SIZE: PropSpec = { name: "vocab_size", kind: "int", default: 32768, min: 1 };
const TIED: PropSpec = { name: "tied", kind: "bool", default: true };

/** lmHead: hidden -> logits; weight tied to embedding => 0 params. */
export const lmHead: LayerSpec = {
  kind: "lmHead",
  label: "LM Head",
  summary: "Projects hidden states to logits (vocab_size); tied weight costs 0 params.",
  inputs: [HIDDEN],
  outputs: [{ name: "logits", label: "Logits" }],
  props: [VOCAB_SIZE, TIED],
  inferShape(props, inputs) {
    const inSh = requireIn(inputs, 0, "lmHead", HIDDEN);
    const vocabSize = numProp("vocab_size", props.vocab_size, VOCAB_SIZE);
    return { ...inSh, d: vocabSize };
  },
  paramCount(props, _out, inputs) {
    const tied = boolProp("tied", props.tied);
    if (tied) return 0;
    const vocabSize = numProp("vocab_size", props.vocab_size, VOCAB_SIZE);
    // Untied head weight is vocab_size x HIDDEN (the input width), NOT the
    // output's vocab-facing d — so this is priced from the hidden input shape.
    if (!inputs?.[0] || typeof inputs[0].d !== "number" || !(inputs[0].d > 0)) {
      throw new Error("lmHead: reference input has no positive d (untied head needs the hidden d)");
    }
    return pEmbed({ vocab: vocabSize, d: inputs[0].d });
  },
};
