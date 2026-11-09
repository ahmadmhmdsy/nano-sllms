// Shared demo prompts (the demo chapters). Authored top-5 predictions per model
// live in each model's YAML demo.preds keyed by prompt id — a model without a
// key gets the generic neutral fallback + big SIMULATED badge.
export interface DemoPrompt {
  id: string;
  label: string;
  toks: string[];
  note: string;
}
export const DEMO_PROMPTS: DemoPrompt[] = [
  {
    id: "france",
    label: "The capital of France is",
    toks: ["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"],
    note:
      "World knowledge is exercised in the middle-band FFNs — a code-trained model is visibly less confident on it (see nano's flat top-5).",
  },
  {
    id: "code",
    label: "def top_k(logits, k):",
    toks: ["def", "Ġtop", "_k", "(", "logits", ",", "Ġk", "):"],
    note:
      "Code structure is learned statistics: newline + indentation after a def signature — the early and middle layers both expect it.",
  },
  {
    id: "fox",
    label: "The quick brown fox",
    toks: ["The", "Ġquick", "Ġbrown", "Ġfox"],
    note:
      "An idiom is memorized statistics: the induction pattern fires — “last time, after this phrase came…”.",
  },
];
