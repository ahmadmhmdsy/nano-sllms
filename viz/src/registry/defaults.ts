// Shared DEFAULTS prose (plan §4): every per-model YAML only overrides where the
// model genuinely differs. Facts keys map to BlockInspector sections.
import type { ModelSpec } from "../schema/spec";

export const DEFAULT_FACTS: NonNullable<ModelSpec["facts"]> = {
  tokensNote:
    "The model never sees letters. Text is chopped into subword tokens (Ġ marks a leading space, Ċ a newline), and each token ID points at a row of the embedding table.",
  embed:
    "Token ID → row lookup in a learned matrix. That vector is the token's initial meaning — a point in space where similar tokens sit close together. It knows single tokens, not combinations; context comes later.",
  embedTech:
    "Embedding lookup · vocab rows × d_model · no bias · feeds embed_tokens.",
  lnorm:
    "Before attention, before the FFN, and once at the very end, each token's vector is rescaled to a standard length and multiplied by a learned per-dimension gain. It stores no knowledge — but this pre-norm placement is what allows deep stacks to train cleanly.",
  add:
    "Attention and FFN never overwrite the token vector; they add a small update onto it. The vector is a running document that each layer annotates — grammar notes early, meaning notes in the middle, prediction notes late.",
  addTech:
    "x ← x + F(norm(x)) · the residual norm grows with depth · the reason pre-norm + residual trains deeper than post-norm.",
};

// RMSNorm technical line (needs eps + layer count → computed, not stored):
export function normTech(eps: string, layers: number, d: number): string {
  return (
    "x / rms(x) · γ · rms = √(mean(x²)+" + eps + ") · pre-norm ×2 per layer + final norm · " +
    ((2 * layers + 1) * d).toLocaleString("en-US") + " params in total."
  );
}
export function finalNormPlain(layers: number): string {
  return (
    "After the last block, each token vector is normalized one more time so the output head always sees a consistent scale — plumbing, but necessary after " +
    layers + " layers of accumulated edits."
  );
}
export function finalNormTech(eps: string, d: number): string {
  return "same RMSNorm (eps " + eps + ", gain " + d + ") · output [B,L," + d + "] → lm_head.";
}
export function logitsPlain(vocab: number): string {
  return (
    "Raw logits over " + vocab.toLocaleString("en-US") + " tokens become probabilities via softmax. Greedy takes the top one; temperature / top-k sampling adds variety. The winner is appended to the text and the entire pass runs again — one token per forward pass."
  );
}
export function logitsTech(): string {
  return "p = softmax(z/T) · no softmax inside the model — training applies cross-entropy over shifted labels.";
}
