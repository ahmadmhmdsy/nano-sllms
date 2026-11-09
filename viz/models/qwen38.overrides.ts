// Escape-hatch example (plan G2/D3): per-model FACT overrides only — no logic.
// Every key inside facts REPLACES the generic default for this model.
export default {
  facts: {
    embed:
      "Token ID → row lookup in a learned 248,320 × 2,560 matrix. NOT tied here — a separate output head mirrors it (two ~0.64B tables), and the rows also cover image/video tokens: the model is multimodal.",
    lmhead:
      "The newest token's 2,560-number summary is dotted against every row of a SEPARATE output matrix — 248,320 dot products, one per possible token (untied, also covering image/video tokens).",
    ffn:
      "Here the knowledge base is SHARDED instead of one dense block. A tiny router activates 10 of 512 expert SwiGLU key–value memories plus one always-on shared expert; their combined update is added to the token. 24,576 experts across the stack hold ~97% of all parameters.",
  },
};
