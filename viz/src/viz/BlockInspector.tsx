// Block inspector — port of COMPS/showDetail. Plain/Technical seg toggle,
// knowledge-heat meter, GQA pairing drawing, honest notes.
import { useState, type ReactNode } from "react";
import type { ModelSpec } from "../schema/spec";
import { bandOf, heat } from "../engine/depth";
import { fmtP, fmtB, pTotal, pEmbed, pAttentionDense, pFFNDense, pMixerLinear, pMixerFull, pMoeLayer } from "../engine/params";
import { factoryDefaults } from "../registry/models";
import * as dense from "./kinds/denseGqaMoe";
import * as hyb from "./kinds/hybridLinearMoe";

export interface SelRef {
  model: string;
  block: string;
  layer?: number;
}

interface Entry {
  title: string;
  tagline: string;
  plain: string;
  tech: string;
}

function isLybrid(m: ModelSpec) {
  return m.archKind === "hybridLinearMoe";
}

function headDim(l: number, of: number) {
  return Math.floor(l / of);
}

export function entryFor(spec: ModelSpec, block: string, layer: number | undefined): Entry | null {
  const f = factoryDefaults(spec);
  const M = spec.shape;
  const L = M.layers;
  const hybrid = isLybrid(spec);
  const full = hybrid ? (layer ?? 1) % (spec.hybrid?.fullEvery ?? 4) === 0 : false;
  const b = layer ? bandOf(spec.bands, layer) : "";
  const total = pTotal(spec);

  switch (block) {
    case "tokens":
      return {
        title: "Tokenization — text → IDs",
        tagline: "The doorway: words become subword pieces, pieces become integers.",
        plain:
          "The model never sees letters. Text is chopped into subword tokens by the " +
          spec.tok +
          ", and each token ID points at a row of the embedding table. " +
          (f.tokensNote ?? "") +
          " Different vocabularies (here " +
          fmtP(M.vocab) +
          "), so the same sentence can cost a different number of tokens per model.",
        tech:
          "BPE tokenizer · vocab " + fmtP(M.vocab) + ". Ġ = leading space, Ċ = newline. No trainable weights — a pure lookup that feeds embed_tokens.",
      };
    case "embed":
      return {
        title: "Embedding table",
        tagline: "Where meaning begins: one learned vector per vocabulary row.",
        plain:
          "Token ID → row lookup in a learned " + fmtP(M.vocab) + " × " + M.d + " matrix (" +
          fmtP(pEmbed(M)) + " parameters). " + f.embed! +
          (hybrid ? "" : " TIED: this same matrix is reused as the output head, so reading tokens in and writing predictions out share one geometry."),
        tech: (f.embedTech ?? "").replace(
          /__CALC__/g,
          String(pEmbed(M))
        ) + (hybrid ? "" : " · tied with lm_head.weight — a tied model's safetensors omit lm_head.") +
          " · " + ((100 * pEmbed(M)) / total).toFixed(1) + "% of the model.",
      };
    case "layer": {
      const role =
        b === "early"
          ? "Early layers mostly handle surface form: spelling, local grammar, short-range copying. Attention looks at nearby tokens; the FFN sharpens token-level patterns."
          : b === "mid"
          ? "Middle layers are the knowledge hotspot: FFN key–value memories fire on combinations of features and inject factual, semantic content. Most of what the model knows is exercised here."
          : "Late layers assemble the answer: attention gathers what the stack accumulated, features become prediction-shaped, and the representation converges on the next token.";
      return {
        title:
          (hybrid ? "block " : "decoder block · layer ") +
          (layer ?? 0) +
          (hybrid ? (full ? " · full attn" : " · GDN") : ""),
        tagline: "One refinement pass on the residual stream: route information, then recall knowledge.",
        plain: "Layer " + layer + " of " + L + " — " + b + " band. " + role,
        tech: hybrid
          ? "x = x + Mixer(RMSNorm(x)); x = x + MoE(RMSNorm(x)). Mixer — " + (full ? "gated full attention (" + M.heads + "Q/" + M.kv + "KV, head " + (spec.hybrid?.fullMixer?.headDim ?? 256) + ", partial RoPE " + (spec.hybrid?.fullMixer?.partialRotary ?? "") + ")" : "Gated DeltaNet linear attention (" + spec.hybrid?.linearMixer?.kHeads + " K / " + spec.hybrid?.linearMixer?.vHeads + " V heads ×" + spec.hybrid?.linearMixer?.headDim + ", conv-" + spec.hybrid?.linearMixer?.conv + "). ") + "FFN: sparse MoE " + fmtB(pMoeLayer(M, spec.hybrid!.experts!)) + "/layer. Band: " + b + " · eps " + M.eps + "."
          : "x = x + Attn(RMSNorm(x)); x = x + SwiGLU(RMSNorm(x)). Stream [B,L," + M.d + "]; per-layer params " + fmtP(pAttentionDense(M) + pFFNDense(M)) + ". Band: " + b + " · RoPE θ " + spec.shape.theta + " · eps " + M.eps + ".",
      };
    }
    case "attn":
      if (hybrid && !full) {
        return {
          title: "GDN linear attention — the state mixer",
          tagline: "Decides which information matters, by compressing the past into a fixed-size state.",
          plain:
            "This is a Gated DeltaNet — a LINEAR attention layer: instead of scoring every past token, each token reads and writes a fixed-size recurrent state. Constant memory per token and no KV cache to grow — the engineered answer to " + spec.shape.ctx.replace(/[^0-9,]/g, "") + "-token context. Three of every four layers look like this.",
          tech:
            "GDN: in_proj " + M.d + "×(" + spec.hybrid!.linearMixer!.kHeads * spec.hybrid!.linearMixer!.headDim + "+" + spec.hybrid!.linearMixer!.qHeads * spec.hybrid!.linearMixer!.headDim + "+" + spec.hybrid!.linearMixer!.vHeads * spec.hybrid!.linearMixer!.headDim + ") · conv1d(k=" + spec.hybrid!.linearMixer!.conv + ") · SiLU·dt gate · state " + spec.hybrid!.linearMixer!.kHeads * spec.hybrid!.linearMixer!.headDim + "×" + spec.hybrid!.linearMixer!.vHeads * spec.hybrid!.linearMixer!.headDim + " · out_proj → " + M.d + " · ε≈" + fmtB(pMixerLinear(M, spec.hybrid!.linearMixer!)) + " params/layer · no KV cache (O(1) state/token) · sizes computed from the recipe; indexer/conv variants folded into extras.",
        };
      }
      return {
        title: hybrid ? "Gated attention (1-in-4) — the exact mixer" : "GQA attention — the router",
        tagline: "Decides which earlier tokens matter right now, and copies their information over.",
        plain:
          "Each token emits a query ('what am I looking for?') plus keys and values ('what do I offer?'). Scores against every EARLIER token only — the causal mask — decide what gets blended in. Attention is information ROUTING, not storage. GQA lets " + M.heads + " query heads share just " + M.kv + " key/value heads — a " + headDim(M.heads, M.kv) + "× smaller KV cache than full multi-head attention." +
          (hybrid ? " One layer in four is exact: GQA " + M.heads + "Q/" + M.kv + "KV head_dim " + (spec.hybrid?.fullMixer?.headDim ?? 256) + ", only " + (spec.hybrid?.fullMixer?.partialRotary ?? "") + " of dims rotary-embedded, plus a lightweight indexer that pre-selects tokens worth scoring." : ""),
        tech: hybrid
          ? "full attn: Q " + M.d + "×" + (M.heads * (spec.hybrid?.fullMixer?.headDim ?? 256)) + " · K,V " + M.d + "×" + (M.kv * (spec.hybrid?.fullMixer?.headDim ?? 256)) + " · O " + (M.heads * (spec.hybrid?.fullMixer?.headDim ?? 256)) + "×" + M.d + " · ≈" + fmtB(pMixerFull(M, spec.hybrid?.fullMixer?.headDim ?? 256)) + " params/layer · partial rotary " + (spec.hybrid?.fullMixer?.partialRotary ?? "") + "."
          : "WQ " + M.d + "×" + M.d + " · WK,WV " + M.d + "×" + M.kv * headDim(M.d, M.heads) + " · WO " + M.d + "×" + M.d + " · " + headDim(M.heads, M.kv) + " Q/KV group · " + fmtP(pAttentionDense(M)) + " params/layer · KV cache " + (2 * M.kv * headDim(M.d, M.heads)) + " fp16 numbers/token · " + (spec.shape.attentionImpl ?? "SDPA") + " kernel.",
      };
    case "ffn":
      return {
        title: hybrid ? "Sparse MoE — the sharded knowledge base" : "SwiGLU FFN — the knowledge base",
        tagline: "The parameter-heavy block: the model's key–value memory.",
        plain: f.ffn! +
          (hybrid ? "" : " With " + fmtP(M.ffn) + " slots per layer × " + L + " layers, this is where most parameters live: " + ((100 * pFFNDense(M)) / total).toFixed(0) + "% of this model."),
        tech: hybrid
          ? "router: " + M.d + "→" + spec.hybrid!.experts!.total + " · each expert SwiGLU " + M.d + "→" + fmtP(spec.hybrid!.experts!.inter) + "→" + M.d + " = " + fmtP(3 * M.d * spec.hybrid!.experts!.inter) + " params · " + fmtB(pMoeLayer(M, spec.hybrid!.experts!)) + "/layer · " + ((100 * L * pMoeLayer(M, spec.hybrid!.experts!)) / total).toFixed(1) + "% of all parameters."
          : "gate,up: " + M.d + "→" + fmtP(M.ffn) + " · down: " + fmtP(M.ffn) + "→" + M.d + " · act = SiLU(gate) ⊙ up · " + fmtP(pFFNDense(M)) + " params/layer = " + ((100 * pFFNDense(M)) / total).toFixed(1) + "% of total · input is RMSNorm(x).",
      };
    case "lnorm":
      return {
        title: "RMSNorm — the stabilizer",
        tagline: "Rescales every token vector so deep stacks stay stable.",
        plain: f.lnorm! + " Just " + M.d + " volume knobs per norm — no knowledge.",
        tech:
          "x / rms(x) · γ · rms = √(mean(x²)+" + spec.shape.eps + ") · pre-norm ×2 per layer + final norm · " +
          fmtP((2 * L + 1) * M.d) + " params in total.",
      };
    case "add":
      return {
        title: "Residual add — the highway",
        tagline: "The connection that carries everything: layers only ADD edits.",
        plain: f.add!,
        tech: f.addTech!,
      };
    case "finalnorm":
      return {
        title: "Final RMSNorm",
        tagline: "The last rescale before reading out.",
        plain:
          "After the last block, each token vector is normalized one more time so the output head always sees a consistent scale — plumbing, but necessary after " + L + " layers of accumulated edits.",
        tech: "same RMSNorm (eps " + spec.shape.eps + ", gain " + M.d + ") · output [B,L," + M.d + "] → lm_head.",
      };
    case "lmhead":
      return {
        title: hybrid ? "LM head — separate (untied)" : "LM head — tied to the embedding",
        tagline: "The embedding table read in reverse: vector → vocabulary scores.",
        plain: f.lmhead! +
          (hybrid ? "" : " TIED: the SAME matrix used for input embeddings is reused, saving " + fmtP(pEmbed(M)) + " parameters and forcing input and output geometry to agree."),
        tech: "lm_head = Linear(" + M.d + " → " + fmtP(M.vocab) + ", bias=False)" + (spec.shape.tied ? ", weight = embed_tokens.weight (tied)" : ", untied — separate output matrix") + " · no softmax inside.",
      };
    case "logits":
      return {
        title: "Softmax → next token",
        tagline: "Scores become probabilities; sampling picks the future.",
        plain:
          "Raw logits over " + fmtP(M.vocab) + " tokens become probabilities via softmax. Greedy takes the top one; temperature / top-k sampling adds variety. The winner is appended and the entire pass runs again — one token per forward pass.",
        tech: "p = softmax(z/T) · no softmax inside the model — training applies cross-entropy over shifted labels.",
      };
    case "band":
      return {
        title: "The middle band — knowledge hotspot",
        tagline: "One bar per middle layer; brighter = more knowledge traffic (rule-of-thumb).",
        plain: f.band ??
          "Depth specializes. In this middle stretch, FFN key–value memories fire hardest: facts, idioms, code patterns, 'how the world is wired'. Click an individual bar to inspect that exact layer. Early layers around it handle surface form; late layers assemble the prediction." +
            (hybrid ? " Here the memory is sharded: 512 routed experts per layer act as many small key–value stores; the router picks a team of 10 per token." : ""),
        tech:
          "bands — early " + spec.bands.early.join("–") + ", middle " + spec.bands.mid.join("–") + ", late " + spec.bands.late.join("–") + " · heat = triangular ramp peaking at the mid-band center — a visualization of the Geva/ROME consensus, not a measured value.",
      };
    case "logits-x":
      return null;
    default:
      return null;
  }
}

export function GQADiagram({ spec }: { spec: ModelSpec }) {
  const M = spec.shape;
  const rows = Math.ceil(M.heads / 2);
  const rowH = 24, top = 8, W = 360;
  const kvY = (k: number) => top + ((k + 0.5) * (rows * rowH)) / M.kv;
  const cells: ReactNode[] = [];
  for (let q = 0; q < M.heads; q++) {
    const col = Math.floor(q / rows), row = q % rows;
    const x = 20 + col * 96, yy = top + row * rowH;
    const kv = Math.floor(q / (M.heads / M.kv));
    cells.push(
      <g key={"q" + q}>
        <rect x={x} y={yy} width={56} height={18} rx={4} fill="rgba(224,164,88,.18)" stroke="#e0a458" />
        <text x={x + 28} y={yy + 13} textAnchor="middle" fontSize={9.5} fill="#e0c9a0" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>Q{q}</text>
        <line x1={x + 56} y1={yy + 9} x2={230} y2={kvY(kv)} stroke="#7fc8a9" opacity={0.6} />
      </g>
    );
  }
  for (let k = 0; k < M.kv; k++) {
    const yy = kvY(k) - 9;
    cells.push(
      <g key={"k" + k}>
        <rect x={230} y={yy} width={110} height={18} rx={4} fill="rgba(127,200,169,.14)" stroke="#7fc8a9" />
        <text x={285} y={yy + 13} textAnchor="middle" fontSize={9.5} fill="#a9d9c2" style={{ fontFamily: "ui-monospace,Consolas,monospace" }}>K/V head {k}</text>
      </g>
    );
  }
  return <svg viewBox={"0 0 " + W + " " + (rows * rowH + 26)} style={{ width: "100%", height: "auto" }}>{cells}</svg>;
}

export function BlockInspector({
  spec,
  sel,
}: {
  spec: ModelSpec;
  sel: SelRef | null;
}) {
  const [mode, setMode] = useState<"plain" | "tech">("plain");
  if (!sel || sel.model !== spec.id) {
    return (
      <div>
        <h3 style={{ margin: "6px 0 4px" }}>Pick a block</h3>
        <p style={{ color: "var(--dim)", fontSize: 14 }}>
          Click any shape in a diagram — token chips, the embedding table, a whole decoder layer, the attention or FFN step inside it, the knowledge band, the final norm, or the output head.
        </p>
      </div>
    );
  }
  const e = entryFor(spec, sel.block, sel.layer);
  if (!e) {
    return (
      <div>
        <h3 style={{ margin: "6px 0 4px" }}>Unknown block “{sel.block}”</h3>
        <p style={{ color: "var(--dim)" }}>Registered blocks: tokens, embed, layer, attn, ffn, lnorm, add, band, finalnorm, lmhead, logits.</p>
      </div>
    );
  }
  const hybrid = isLybrid(spec);
  const b = sel.layer ? bandOf(spec.bands, sel.layer) : null;
  const ht = sel.layer ? heat(spec.bands, sel.layer) : 0;
  const showGQA = sel.block === "attn" && (!hybrid || (sel.layer ? sel.layer % (spec.hybrid?.fullEvery ?? 4) === 0 : false));
  const showGdnNote = sel.block === "attn" && hybrid && !(sel.layer ? sel.layer % (spec.hybrid?.fullEvery ?? 4) === 0 : true);
  return (
    <div>
      <div className="crumb">
        {spec.name}
        {sel.layer ? " · layer " + sel.layer : ""} · {e.title}
      </div>
      <h3>{e.title}</h3>
      <p className="tag">{e.tagline}</p>
      <div className="seg">
        <button className={mode === "plain" ? "on" : ""} onClick={() => setMode("plain")}>Plain</button>
        <button className={mode === "tech" ? "on" : ""} onClick={() => setMode("tech")}>Technical</button>
      </div>
      {mode === "plain" ? (
        <p style={{ fontSize: 14.5 }}>{e.plain}</p>
      ) : (
        <p className="mono" style={{ fontSize: 12.5, lineHeight: 1.7, color: "#b8bcc4" }}>{e.tech}</p>
      )}
      {b && (
        <div className="meter">
          knowledge heat at this depth (rule-of-thumb)
          <div className="bar">
            <i style={{ width: (ht * 100).toFixed(0) + "%", background: b === "mid" ? "#a78bdc" : b === "early" ? "#56c8d8" : "#e0a458" }} />
          </div>
          <div style={{ marginTop: 4 }}>{b} band · layer {sel.layer} / {spec.shape.layers}</div>
        </div>
      )}
      {showGdnNote && (
        <div className="note">
          Linear (GDN) mixer — no attention grid exists: a fixed-size recurrent state replaces the token-to-token matrix, and full attention only returns every {spec.hybrid?.fullEvery ?? 4}th layer.
        </div>
      )}
      {showGQA && !showGdnNote && (
        <div className="note">
          Causal mask: token i may look only at tokens j ≤ i. RoPE (θ {spec.shape.theta}) bakes position into Q/K — relative distance, zero extra parameters. GQA pairing:
          <GQADiagram spec={spec} />
        </div>
      )}
      {sel.block === "embed" && !hybrid && (
        <div className="note">This table is also the OUTPUT head (tied weights) — the only component that exists in two places at once.</div>
      )}
      {sel.block === "band" && (
        <div className="note">Same depth-banding shows up across models — a property of the shared recipe, not of one training run.</div>
      )}
    </div>
  );
}
void dense; void hyb;