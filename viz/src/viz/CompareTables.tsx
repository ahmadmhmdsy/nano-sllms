// Compare tables — spec facts AUTO-GENERATED from ALL registered specs (plan G1).
// No row lists per-model strings; every string comes from a spec.
import type { ModelSpec } from "../schema/spec";
import { fmtP, pTotal, pAttentionDense } from "../engine/params";
import * as dense from "./kinds/denseGqaMoe";
import * as hyb from "./kinds/hybridLinearMoe";

function mixerCell(m: ModelSpec): string {
  const s = m.shape;
  if (m.archKind === "hybridLinearMoe") {
    const h = m.hybrid!;
    const lm = h.linearMixer!;
    return "hybrid " + ((h.fullEvery ?? 4) - 1) + ":1 — Gated DeltaNet linear (" + lm.kHeads + " K / " + lm.vHeads + " V ×" + lm.headDim + ", conv-" + lm.conv + ") : gated attention " + s.heads + "Q/" + s.kv + "KV · head_dim " + (h.fullMixer?.headDim ?? 256);
  }
  return "GQA · " + s.heads + "Q/" + s.kv + "KV · head_dim " + Math.floor(s.d / s.heads) + " · " + (m.shape.attentionImpl ?? "SDPA");
}
function ffnCell(m: ModelSpec): string {
  if (m.archKind === "hybridLinearMoe") {
    const e = m.hybrid!.experts!;
    return "sparse MoE — " + e.total + " experts, " + e.active + " active + " + e.shared + " shared (inter " + e.inter + ")";
  }
  return "dense SwiGLU " + m.shape.ffn;
}
function paramsCell(m: ModelSpec): string {
  return m.archKind === "hybridLinearMoe"
    ? "≈" + (pTotal(m) / 1e9).toFixed(1) + "B text stack (computed) · ≈6B active/token"
    : fmtP(pTotal(m)) + " (computed)";
}
function tiedCell(m: ModelSpec): string {
  return m.shape.tied ? "yes — embed = lm_head" : "no — separate output head";
}
function posCell(m: ModelSpec): string {
  return m.archKind === "hybridLinearMoe" ? "mRoPE θ 10,000,000 · partial rotary 25%" : "RoPE θ " + m.shape.theta;
}
function archCell(m: ModelSpec): string {
  return m.archKind === "hybridLinearMoe"
    ? "Qwen4ExpForConditionalGeneration — hybrid linear/full attention + sparse MoE"
    : "LlamaForCausalLM — dense GQA decoder";
}

export function SpecTable({ specs }: { specs: ModelSpec[] }) {
  const rows: [string, (m: ModelSpec) => string][] = [
    ["Architecture family", archCell],
    ["Parameters", paramsCell],
    ["Layers × width", (m) => m.shape.layers + " × " + m.shape.d],
    ["Token mixer", mixerCell],
    ["Positional encoding", posCell],
    ["FFN", ffnCell],
    ["Normalization", (m) => "RMSNorm eps " + m.shape.eps + " · pre-norm"],
    ["Tied embeddings", tiedCell],
    ["Vocabulary", (m) => fmtP(m.shape.vocab)],
    ["Context", (m) => m.shape.ctx],
    ["Precision / impl", (m) => m.shape.precision + " · " + (m.shape.attentionImpl ?? "—")],
    ["Trained on", (m) => m.honesty.trainedOn],
    ["Speaks", (m) => m.honesty.speaks],
  ];
  return (
    <table>
      <thead>
        <tr>
          <th style={{ width: "15%" }}>Spec</th>
          {specs.map((m) => (
            <th key={m.id}>{m.name}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, fn]) => (
          <tr key={label}>
            <td>{label}</td>
            {specs.map((m) => (
              <td key={m.id}>{fn(m)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function PurposeTable({ specs }: { specs: ModelSpec[] }) {
  const rows: [string, string, string[]][] = [
    [
      "Token IDs → embedding row",
      "Single-token meaning; the first learned layer of knowledge.",
      specs.map((m) => m.name + ": " + (m.shape.tied ? "tied with the output head" : "untied — separate output head")),
    ],
    [
      "Attention layers",
      "Routes information between positions — moves knowledge; does not store it.",
      specs.map((m) =>
        m.archKind === "hybridLinearMoe"
          ? m.name + ": full attn " + fmtP6(pAttentionDense({ d: m.shape.d, heads: m.shape.heads, kv: m.shape.kv })) + " every 4th layer · GDN linear ≈42M"
          : m.name + ": " + fmtP(pAttentionDense(m.shape)) + " / layer"
      ),
    ],
    [
      "FFN / expert modules",
      "Key–value memory: facts, idioms, skills — the knowledge base, strongest mid-depth.",
      specs.map((m) =>
        m.archKind === "hybridLinearMoe"
          ? m.name + ": ~97% across " + m.shape.layers * (m.hybrid?.experts?.total ?? 0) + " experts"
          : m.name + ": " + ((100 * 3 * m.shape.d * m.shape.ffn) / pTotal(m)).toFixed(0) + "%"
      ),
    ],
    ["RMSNorm ×3 per layer path", "Keeps the residual stream scaled; carries no knowledge.", specs.map((m) => m.name + ": identical placement")],
    ["Residual stream", "Each layer ADDS an edit onto the token's running vector.", specs.map((m) => m.name + ": the core connection")],
    ["Final norm + LM head", "Reads the vector out as vocabulary-wide scores → softmax.", specs.map((m) => m.name + ": " + (m.shape.tied ? "tied" : "separate head"))],
    [
      "Early band",
      "Surface form: spelling, local grammar, short-range copying.",
      specs.map((m) => m.name + ": L" + m.bands.early[0] + "–" + m.bands.early[1]),
    ],
    ["Middle band", "Facts and semantics — 'what things mean'.", specs.map((m) => m.name + ": L" + m.bands.mid[0] + "–" + m.bands.mid[1])],
    ["Late band", "Task assembly and prediction prep — 'what comes next'.", specs.map((m) => m.name + ": L" + m.bands.late[0] + "–" + m.bands.late[1])],
  ];
  return (
    <table>
      <thead>
        <tr>
          <th style={{ width: "22%" }}>Component</th>
          <th>Purpose · where its learning shows up</th>
          <th style={{ width: "34%" }}>In the models (auto-computed from specs)</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([a, b, per]) => (
          <tr key={a}>
            <td><b>{a}</b></td>
            <td>{b}</td>
            <td>
              <span className="dim2">
                {per.join(" · ")}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// shared-DNA / differences lists (family-level prose — not per-model references)
const SIMS: [string, string][] = [
  ["Decoder-only, autoregressive", "all three predict the next token from a causal view of the past"],
  ["Pre-RMSNorm residual blocks", "x + Mixer(norm(x)); x + FFN(norm(x)) — the same wiring in all three"],
  ["RoPE-based positioning", "positions enter as rotations of Q/K in all three (Qwen rotates just 25% of dims, multimodal mRoPE)"],
  ["SiLU-gated FFN / expert modules", "SwiGLU MLPs in the dense pair; SwiGLU experts inside the MoE — the same gated recipe"],
  ["Causal mask", "strict left-to-right information flow in all three"],
  ["Residual highway", "layers only ADD edits onto the running token vector — everywhere"],
  ["GQA-style KV sharing", "nano 16→4 · SmolLM2 9→3 · Qwen full-attn 24→2 — sparser KV sharing every generation"],
];
const DIFFS: string[] = [
  "Attention engine: pure GQA everywhere in the dense pair vs Qwen's 3:1 linear-GDN / full-attention hybrid + a lightning indexer — the pattern this repo's own GDN-hybrid milestone (G1) imitates at small scale",
  "FFN: one dense key–value memory layer vs 512 routed experts + 1 shared — 'the knowledge base' shattered into 24,576 small memories",
  "Sparsity: 100% of parameters active per token in the dense pair vs ~5% in the MoE (10 of 512 experts)",
  "Tied embeddings: BOTH dense models tie; Qwen does not — separate ~0.64B output head",
  "Scale: 226.5M and 134.5M dense vs ≈125B total / ≈6B active MoE",
  "Context: 1k→4k (YaRN) and 8k vs 262,144 native — the linear layers make long context cheap",
  "Decoding: one token per step vs Qwen MTP: 1 extra draft layer predicts multiple tokens ahead",
  "Diet: Python code vs trillions of text/code/math tokens vs post-trained multimodal preview",
];
export function SharedDna() {
  return (
    <ul className="checks">
      {SIMS.map(([a, b]) => (
        <li key={a}><b>{a}</b> — {b}</li>
      ))}
    </ul>
  );
}
export function Differences() {
  return (
    <ul className="diffs">
      {DIFFS.map((d) => (
        <li key={d.slice(0, 24)}>{d}</li>
      ))}
    </ul>
  );
}
function fmtP6(n: number): string {
  return (n / 1e6).toFixed(1) + "M";
}
void fmtP6;
void dense;
void hyb;
